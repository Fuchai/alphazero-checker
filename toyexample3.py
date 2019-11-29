from time import sleep
import random
import queue
from queue import Empty
from threading import Condition, Lock, Semaphore
import threading
# in this example, I would like to use queue, not concurrent.futures.ThreadPoolExecutor
# this is because the GPU will need to execute by batch nodes, while job submission is by node
# executor will only process one node at a time

class GPU:
    def process(self, edges):
        vals=[]
        for edge in edges:
            vals.append(edge.val+1)
        sleep(1)
        # print("processed", str(len(edges)), "items")
        return vals

def process_agent(gpu, queue):
    while True:
        edges=[]
        last_batch=False
        for i in range(8):
            try:
                edge= queue.get_nowait()
                if isinstance(edge, Sentinel):
                    last_batch=True
                    print("Sentinel received. GPU will process this batch and terminate afterwards")
                else:
                    edges.append(edge)
            except Empty:
                pass
        if len(edges)!=0:
            # batch process
            vals=gpu.process(edges)
            for edge, val in zip(edges, vals):
                edge.val=val
                queue.task_done()
                edge.from_node.lock.release()
            if last_batch:
                queue.task_done()
                print("Queue task done signal sent. Queue will join. Thread may still be running.")
                return
        else:
            sleep(0.1)

class Sentinel:
    def __init__(self):
        self.val="STOP"

class Node:
    def __init__(self, degree):
        self.lock=Semaphore()
        self.edges=[]
        for i in range(degree):
            self.edges.append(Edge(self))

    def putall(self, queue):
        for edge in self.edges:
            self.lock.acquire(blocking=False)
            queue.put(edge)
            # print("submitted an edge")

class Edge:
    def __init__(self, from_node):
        self.val=0
        self.from_node=from_node

class Tree:
    def __init__(self, gpu_execution_queue):
        # like nodes
        self.queue=gpu_execution_queue
        self.nodes=[]
        self.degree=13
        for i in range(random.randint(5,20)):
            self.nodes.append(Node(self.degree))
        self.gpu=GPU()
        self.epoch=random.randint(10,40)

    def run(self):
        for i in range(self.epoch):
            # selection
            # some computation, but not expensive
            sleep(0.1)
            select_idx=random.randint(0,len(self.nodes)-1)
            selected_node=self.nodes[select_idx]
            selected_node.lock.acquire(blocking=True)
            sleep(0.1)
            # work on the selected node, acquire blocks until the expansion finishes
            selected_node.lock.release()

            # expansion
            expand_idx=random.randint(0,len(self.nodes)-1)
            # processing is slow
            # the next iteration might not choose the same index, so it is not necessary to wait for it to finish
            # however, if a node selected is being processed, you must wait for it to finish
            selected_node=self.nodes[expand_idx]
            sleep(0.1)
            selected_node.putall(self.queue)
            # # ret=self.gpu.process(selected_node)
            # selected_node.val=ret
            # print(ret)
        self.queue.put(Sentinel())
        print("Terminal sentinel is put on queue")

    def validation(self):
        incre=0
        for node in self.nodes:
            for edge in node.edges:
                incre+=edge.val
        if incre==self.epoch*self.degree:
            print ("NO RACE CONDITION OBSERVED, expected:", self.epoch*self.degree, "actual: ", incre)
        else:
            print("!!!!!!!!RACE CONDITION!!!!!!")
            print("incre: ", incre)
            print("expected: ", self.epoch*self.degree)
            assert(incre==self.epoch*self.degree)


def stress_test():
    for i in range(100):
        gpu_execution_queue=queue.Queue(maxsize=256)
        tree = Tree(gpu_execution_queue)
        t = threading.Thread(target=process_agent, args=(tree.gpu, gpu_execution_queue))
        t.start()
        tree.run()
        gpu_execution_queue.join()
        print("Queue has joined")
        t.join()
        print("Thread has joined")
        try:
            tree.validation()
        except AssertionError:
            raise
        print("Successful termination")
        print("Queue empty:",gpu_execution_queue.empty())

if __name__ == '__main__':
    stress_test()