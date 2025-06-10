# In the name of God
from typing import List, Any, Tuple

import numpy as np
import simpy
import random
import matplotlib.pyplot as plt
from functools import cmp_to_key

from scipy.optimize import minimize

figures = {}

MAX_NUMBER_OF_FRAMES = 100000

pid = 0
debug = False

linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'dashdot', ':', '--']


class Packet:
    def __init__(self, pid, size, qn):
        self.pid = pid
        self.size = size
        self.enqueue_time = 0
        self.dequeue_time = 0
        self.process_time = 0
        self.qn = qn


class Queue:
    packets: list[Packet]

    def __init__(self, arrival_rate, mean_packet_size, delay_threshold=0, drop_threshold=np.inf, loss_rate=0):
        self.arrival_rate = arrival_rate
        self.mean_packet_size = mean_packet_size

        self.delay_threshold = delay_threshold
        self.drop_threshold = drop_threshold
        self.loss_rate = loss_rate

        self.rn = [0]
        self.yn = [0]
        self.un = [0]

        self.packets = []
        self.delays = []

        self.drops = [0]
        self.non_violations = [0]

    def enqueue(self, packet):
        self.packets.append(packet)

    def dequeue_or_drop(self, time) -> Packet | None:
        p = self.packets.pop(0)
        p.dequeue_time = time
        qdelay = p.dequeue_time - p.enqueue_time
        # self.delays.append(qdelay)
        if qdelay > self.drop_threshold:
            self.drops[-1] = self.drops[-1] + 1
            return None
        else:
            self.non_violations[-1] = self.non_violations[-1] + 1
            return p

    def process_frame(self):
        self.yn.append(
            max(0, self.yn[-1] + sum(self.delays) - (len(self.delays) * self.rn[-1])))

        total_frame_packet_count = (self.drops[-1] + self.non_violations[-1])
        self.un.append(
            max(0, self.un[-1] + (self.drops[-1] - self.loss_rate * total_frame_packet_count))
        )

        self.delays = [])):
            priorities[i] = delays[i][0]
        return priorities

    def calculate_cost(self, priorities):

        a = 0
        b = 0
        for qn in range(len(self.queues)):
            q = self.queues[qn]
            # Σ λn E [Sn]
            a += q.arrival_rate * q.mean_packet_size
            # Σ λn E [Sn ^ 2]
            # b += q.arrival_rate * (q.mean_packet_size + q.mean_packet_size ** 2)
            # just for testing
            b += q.arrival_rate * (2 if qn == 0 else 1)

        def w(nj, cost):

            ro = 0
            service_rate = self.get_service_rate(cost)
            for i in range(nj):
                q = self.queues[priorities[i]]
                ro += q.arrival_rate * q.mean_packet_size / service_rate
            q = self.queues[priorities[nj]]
            d = (1 - ro) * (1 - ro - (q.arrival_rate * q.mean_packet_size / service_rate))

            return 0.5 * b / (service_rate ** 2) / d

        def cost_optimization_function(p):

            f = self.x[-1] * a * p / self.get_service_rate(p)

            for qn in range(len(self.queues)):
                q = self.queues[qn]
                nj = list(filter(lambda x: x[1] == qn, priorities.items()))[0][0]
                f += q.yn[-1] * q.arrival_rate * w(nj, p)

            return f

        # result = minimize(cost_optimization_function, x0=[VALID_COSTS[0]],
        #                   bounds=[(min(VALID_COSTS) - 1, max(VALID_COSTS) + 1)], options={'disp': False})
        # self.costs.append(result.x[0])
        # Just for testing
        VALID_COSTS = [16, 25]
        self.costs.append(min(VALID_COSTS, key=lambda x: cost_optimization_function(x)))

    def get_service_time(self, packet, t):
        return packet.size / self.get_service_rate(self.costs[-1])

    @classmethod
    def get_service_rate(self, p):
        return np.sqrt(p)

    def fn(self, rn, qn):
        C = [0.5, 2]
        return C[qn] * rn ** 2

    def y_optimization_function(self, rn, qn, arrival_rate, y, drop_rate):
        return self.V * self.fn(rn, qn) - arrival_rate * y * rn


def packet_generator(env, average_rate, average_size, server, qn):
    global pid, debug
    while len(server.processed_packets) < MAX_NUMBER_OF_FRAMES:
        yield env.timeout(random.expovariate(average_rate))
        # p = Packet(pid + 1, random.expovariate(1 / average_size), qn)
        # Just for testing
        if qn == 0:
            p = Packet(pid + 1, 0.5 if random.random() < 0.8 else 3, qn)
        else:
            p = Packet(pid + 1, 1, qn)
        if debug:
            print('packet ' + str(p.pid) + ' arrived @ ' + str(env.now) + ' q = ' + str(qn))
        server.receive(p, qn)
        pid = pid + 1


def plot_virtual_delays(server, j=0):
    # global i
    # fig = plt.figure()
    for i in range(len(server.queues)):
        plt.subplot(len(server.queues), 1, i + 1)
        plt.title('Queue ' + str(i))
        plt.plot(server.queues[i].yn, linestyle=linestyle_str[j])
    # fig.suptitle('Virtual Delay')


def plot_objective_function(server, delays, An, chunk=5000, k=0):
    if not 'of' in figures.keys():
        figures['of'] = plt.figure()
        plt.title('Objective Function')
    else:
        plt.figure(figures['of'])
    of = np.zeros((1, len(server.costs)))
    for i in delays.keys():
        steps = np.arange(0, stop=len(delays[i]), step=chunk).tolist()
        steps.append(len(delays[i]))
        n = None
    for i in delays.keys():
        steps = np.arange(0, stop=len(delays[i]), step=chunk).tolist()
        steps.append(len(delays[i]))
        n = None
        dn = None
        for j in range(len(steps[:-1])):
            start = steps[j]
            stop = steps[j + 1]
            size = len(delays[i][start:stop])

            mask = np.triu(np.ones((size, size)))
            for u in range(mask.shape[1]):
                if u < window:
                    continue
                mask[0:u - window, u] = 0

            if n is not None:
                ln = n[-1]
                ldn = dn[-1]
                n = np.hstack((n, np.matmul(delays[i][start:stop], mask) + ln))
                dn = np.hstack((dn, np.matmul(An[i][start:stop], mask) + ldn))
            else:
                n = np.matmul(delays[i][start:stop], mask)
                dn = np.matmul(An[i][start:stop], mask)
        Wn = np.divide(n, dn)
        plt.subplot(len(delays.keys()), 1, i + 1)
        plt.title('Queue ' + str(i))
        plt.plot(Wn, linestyle=linestyle_str[k])
    # plt.title('Average Queueing delay')


def plot_drops(server):
    # global i
    if not 'drop' in figures.keys():
        figures['drop'] = plt.figure()
        plt.suptitle('Drops')
    else:
        plt.figure(figures['drop'])

    for i in range(len(server.queues)):
        plt.subplot(len(server.queues), 1, i + 1)
        plt.plot(np.divide(server.queues[i].drops, np.add(server.queues[i].drops, server.queues[i].non_violations)))
        plt.title('Queue ' + str(i))
        violations_number = np.sum(server.queues[i].drops)
        non_violations_number = np.sum(server.queues[i].non_violations)
        print('Queue {:.0f} drop rate = {:.2f}%'.format(i, violations_number / (
                violations_number + non_violations_number) * 100))


def extract_frame_information(server):
    An = {}
    adelays = {}
    delays = {}
    for i in range(len(server.queues)):
        An[i] = []
        adelays[i] = []
        delays[i] = []
        for frame in server.processed_packets:
            f_delays = []
            for p in frame:
                if p.qn != i:
                    continue
                f_delays.append(p.dequeue_time - p.enqueue_time)
            adelays[i].append(np.mean(f_delays) if f_delays else 0)
            delays[i].append(np.sum(f_delays))
            An[i].append(len(f_delays))
    return adelays, delays, An


def plot_average_queueing_delay(delays, An, chunk=5000, k=0):
    if not 'aqd' in figures.keys():
        figures['aqd'] = plt.figure()
        plt.suptitle('Average Queueing delay')
    else:
        plt.figure(figures['aqd'])

    for i in delays.keys():
        steps = np.arange(0, stop=len(delays[i]), step=chunk).tolist()
        steps.append(len(delays[i]))
        n = None
        dn = None
        for j in range(len(steps[:-1])):
            start = steps[j]
            stop = steps[j + 1]
            size = len(delays[i][start:stop])

            if n is not None:
                ln = n[-1]
                ldn = dn[-1]
                n = np.hstack((n, np.matmul(delays[i][start:stop], np.triu(np.ones((size, size)))) + ln))
                dn = np.hstack((dn, np.matmul(An[i][start:stop], np.triu(np.ones((size, size)))) + ldn))
            else:
                n = np.matmul(delays[i][start:stop], np.triu(np.ones((size, size))))
                dn = np.matmul(An[i][start:stop], np.triu(np.ones((size, size))))
        Wn = np.divide(n, dn)
        plt.subplot(len(delays.keys()), 1, i + 1)
        plt.title('Queue ' + str(i))
        plt.plot(Wn, linestyle=linestyle_str[k])


def print_average_cost(server):
    p = 0
    t = 0
    for i in range(len(server.busy_periods)):
        p += server.busy_periods[i][1] * server.costs[i]
        t += server.busy_periods[i][1]
    for i in range(len(server.idle_periods)):
        t += server.idle_periods[i][1]
    print("average cost is " + str(p / t))


def print_units(server: Server):
    frame = 0
    for cost, period, packets in zip(server.costs, server.busy_periods, server.processed_packets):
        total_size = 0
        for p in packets:
            total_size += p.size

        print(f'frame {frame} : utilized resources = {cost * (period[0] - period[1])} vs actual resources {total_size}')
        frame += 1


if __name__ == '__main__':
    # average inter arrival rate and average packet size
    # the inter arrival rate is in us
    # the average packet size is in KB
    generator_parameters = [(1, 1), (2, 1)]
    # delay, drop, loss rate
    queue_parameters = [(0.6, 12, 2e-3), (0.3, 6, 2e-3)]
    cost_threshold = 13.5

    j = 0
    for v in [10, 100, 1000, 10000]:
        env = simpy.Environment()

        server = Server(env, queue_parameters, generator_parameters, cost_threshold)

        for i in range(len(generator_parameters)):
            parameters = generator_parameters[i]
            env.process(packet_generator(env, parameters[0], parameters[1], server, i))
        env.run(until=10000000)

        adelays, delays, An = extract_frame_information(server)
        plot_average_queueing_delay(delays, An, chunk=5000, k=j)
        plot_sliding_delay(delays, An, chunk=5000, window=1000, k=j)
        plot_objective_function(server, delays, An, chunk=5000, k=j)
        print_average_cost(server)
        j += 1
    plot_drops(server)
    for fig in figures.values():
        plt.figure(fig)
        plt.legend(['V=10', 'V=100', 'V=1000', 'V=10000'])
        plt.show()
    # fig.suptitle('Sliding Queueing delay')
    # plot_virtual_delays(server, j)
    # plot_drops(server)
    #
    # adelays, delays, An = extract_frame_information(server)
    #
    # plt.figure()
    # for i in range(len(generator_parameters)):
    #     plt.subplot(len(server.queues), 1, 1)
    #     plt.plot(adelays[i])
    #     plt.title('average frame delay')
    #     plt.legend([0, 1])
    #     plt.subplot(len(server.queues), 1, 2)
    #     plt.plot(An[i])
    #     plt.title('frame size')
    #     plt.legend([0, 1])
    #
    # plot_average_queueing_delay(delays, An)
    # plt.legend([x for x in queue_parameters])
    #
    # # print(server.idle_periods)
    # # print('###############################')
    # # print(server.busy_periods)
    #
    # # print(len(server.queues[0].virtual_delays))
    # # print(len(server.idle_periods))
    # # print(len(server.busy_periods))
    print_average_cost(server)
    #
    # plt.figure()
    # plt.plot(server.costs)
    # plt.title('Costs')
    #
    # # print_units(server)

    # plt.show()
