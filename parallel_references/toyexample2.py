from time import sleep
import random
import queue
from queue import Empty
from threading import Condition, Lock
import threading
# in this example, I would like to use queue, not concurrent.futures.ThreadPoolExecutor
# this is because the GPU will need to execute by batch nodes, while job submission is by node
# executor will only process one node at a time

class GPU:
    def process(self, nodes):
        vals=[]
        for node in nodes:
            vals.append(node.val+1)
        sleep(1)
        print("processed", str(len(nodes)), "items")
        return vals

def process_agent(gpu, queue):
    while True:
        nodes=[]
        last_batch=False
        for i in range(8):
            try:
                node= queue.get_nowait()
                if isinstance(node, Sentinel):
                    last_batch=True
                    print("Sentinel received")
                else:
                    nodes.append(node)
            except Empty:
                pass
        if len(nodes)!=0:
            # batch process
            vals=gpu.process(nodes)
            for node, val in zip(nodes, vals):
                node.val=val
                queue.task_done()
                node.lock.release()
            if last_batch:
                queue.task_done()
                print("last batch terminate")
                return
        else:
            sleep(0.1)

class Sentinel:
    def __init__(self):
        self.val="STOP"

class Node:
    def __init__(self):
        self.val=0
        self.lock=Lock()

class Tree:
    def __init__(self, gpu_execution_queue):
        # like nodes
        self.queue=gpu_execution_queue
        self.nodes=[]
        for i in range(10):
            self.nodes.append(Node())
        self.gpu=GPU()
        self.epoch=20

    def run(self):
        for i in range(self.epoch):
            # some computation, but not expensive
            sleep(0.1)
            idx=random.randint(0,9)
            # processing is slow
            # the next iteration might not choose the same index, so it is not necessary to wait for it to finish
            # however, if a node selected is being processed, you must wait for it to finish
            selected_node=self.nodes[idx]
            selected_node.lock.acquire()
            self.queue.put(selected_node)
            print("submitted")
            # # ret=self.gpu.process(selected_node)
            # selected_node.val=ret
            # print(ret)
        self.queue.put(Sentinel())
        print("terminal sentinel")

    def validation(self):
        incre=0
        for node in self.nodes:
            incre+=node.val
        if incre==self.epoch:
            print ("no race condition")
        else:
            print("race condition")
            assert(incre==self.epoch)


def main1():
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu_executor:
        tree=Tree(gpu_executor)
        tree.run()

def main2():
    gpu_execution_queue=queue.Queue(maxsize=256)
    tree = Tree(gpu_execution_queue)
    t = threading.Thread(target=process_agent, args=(tree.gpu, gpu_execution_queue))
    t.start()
    tree.run()
    gpu_execution_queue.join()
    tree.validation()
    t.join()
    print("Successful termination")

if __name__ == '__main__':
    main2()