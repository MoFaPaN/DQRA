# In the name of God
from typing import List, Any, Tuple

import numpy as np
import simpy
import random
import matplotlib.pyplot as plt
from functools import cmp_to_key

from scipy.optimize import minimize

MAX_NUMBER_OF_FRAMES = 100000

pid = 0
debug = False

linestyle_str = ['solid', 'dotted', 'dashed', 'dashdot', 'solid', 'dotted', 'dashed', 'dashdot', ':', '--']


class Packet:
    def __init__(self, pid, size, cn):
        self.pid = pid
        self.size = size
        self.enqueue_time = 0
        self.dequeue_time = 0
        self.process_time = 0
        self.cn = cn


class Queue:
    packets: list[Packet]

    def __init__(self, arrival_rate, mean_packet_size, delay_threshold=0, drop_threshold=np.inf):
        self.arrival_rate = arrival_rate
        selfpend(packet)

    def dequeue(self, time) -> Packet:
        p = self.packets.pop(0)
        p.dequeue_time = time
        qdelay = (p.dequeue_time - p.enqueue_time)
        self.delays.append(qdelay)
        if qdelay > self.drop_threshold:
            self.violations[-1] = self.violations[-1] + 1
        else:
            self.non_violations[-1] = self.non_violations[-1] + 1
        return p

    def process_frame(self):
        # self.zn.append(
        #     max(0, self.zn[-1] + sum(self.delays) - (len(self.delays) * self.delay_threshold)))
        self.yn.append(
            max(0, self.yn[-1] + sum(self.delays) - (len(self.delays) * self.rn[-1])))
        self.delays = []
        self.violations.append(0)
        self.non_violations.append(0)

    def is_empty(self):
        return len(self.packets) == 0


class Server:
    def __init__(self, env, queue_parameters, class_parameters, cost_threshold):
        self.V = 1000

        self.env = env
        self.queues = []
        self.processed_packets = []
        for (delay_threshold, drop_threshold), (arrival_rate, mean_packet_size) in zip(queue_parameters,
                                                                                       class_parameters):
            self.queues.append(Queue(arrival_rate, mean_packet_size, delay_threshold, drop_threshold))

    def receive(self, packet, qn):
        global debug
        packet.enqueue_time = env.now
        self.queues[qn].enqueue(packet)
        if not self.is_busy:
            self.env.process(self.process())
        elif debug:
            print('packet ' + str(packet.pid) + ' is queued @ q=' + str(qn))

    def process(self):
        global debug
        self.idle_periods.append((self.start_idle_time, self.env.now - self.start_idle_time))
        start_busy_period = self.env.now
        self.is_busy = True

        priorities = self.calculate_priorities()
ocess_frame()
        bp = self.busy_periods[-1][1]
        ip = self.idle_periods[-1][1]
        t = ip + bp
        self.x.append(
            max(0, self.x[-1] + self.costs[-1] * bp - t * self.cost_threshold))

    def pick_packet(self, priorities) -> Packet | None:
        for k in priorities.keys():
            queue = self.queues[priorities[k]]
            if not queue.is_empty():
                return queue.dequeue(self.env.now)
        return None

    def calculate_priorities(self) -> []:
        delays = []
        for i in range(len(self.queues)):
            queue = self.queues[i]
            delays.append((i, queue.yn[-1] / queue.mean_packet_size))
        delays.sort(
            key=cmp_to_key(lambda x, y: 1 if x[1] > y[1] else -1 if y[1] > x[1] else (random.randint(0, 1) - 0.5)),
            reverse=True)
        priorities = {}
        for i in range(len(delays)):
            priorities[i] = delays[i][0]
        return priorities

    def calculate_cost(self, priorities):

        a = 0
        b = 0
        for qn in range(len(self.queues)):
            q = self.queues[qn]
            # Σ λn E [Sn]
            a += q.arrival_rate * q.mean_packet_size
         (0, stop=len(delays[i]), step=chunk).tolist()
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
        of = of + np.array(server.fn(Wn, i))
    plt.title('Objective Function')
    plt.plot(of[0, 1:], linestyle=linestyle_str[k])


def plot_sliding_delay(delays, An, chunk=5000, window=1000, k=0):
    # plt.figure()
    for i in delays.keys():
        steps = np.arange(0, stop=len(delays[i]), step=chunk).tolist()
        steps.append(len(delays[i]))
        n = None
        dn = None
  
    for i in range(len(server.queues)):
        plt.subplot(len(server.queues), 1, i + 1)
        plt.plot(server.queues[i].drops)
        plt.title('Queue ' + str(i))
        violations_number = np.sum(server.queues[i].drops)
        non_violations_nessed_packets:
            f_delays = []
            for p in frame:
                if p.cn != i:
                    continue
                f_delays.append(p.dequeue_time - p.enqueue_time)
            adelays[i].append(np.mean(f_delays) if f_delays else 0)
            delays[i].append(np.sum(f_delays))
            An[i].append(len(f_delays))
    return adelays, delays, An


def plot_average_queueing_delay(delays, An, chunk=5000, k=0):

    # plt.figure()
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
    # plt.title('Average Queueing delay')


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
    # the inter arrival rate is in ms
    # the average packet size is in KB
    generator_parameters = [(1, 1), (2, 1)]
    pdbs = [100, 100]
    queue_parameters = [(0.6, 1.2), (0.3, 0.6)]
    cost_threshold = 13.5

    fig = plt.figure()
    j = 0
    for v in [10, 100, 1000, 10000]:
        env = simpy.Environment()

        server = Server(env, queue_parameters, generator_parameters, cost_threshold)

        for i in range(len(generator_parameters)):
            parameters = generator_parameters[i]
            env.process(packet_generator(env, parameters[0], parameters[1], server, i))
        env.run(until=10000000)

        adelays, delays, An = extract_frame_information(server)
        # plot_average_queueing_delay(delays, An, chunk=5000, k=j)
        plot_sliding_delay(delays, An, chunk=5000, window=1000, k=j)
        j += 1
    plt.legend(['V=10', 'V=100', 'V=1000', 'V=10000'])
    fig.suptitle('Sliding Queueing delay')
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
    # print_average_cost(server)
    #
    # plt.figure()
    # plt.plot(server.costs)
    # plt.title('Costs')
    #
    # # print_units(server)

    plt.show()
