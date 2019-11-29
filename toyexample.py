from time import sleep
import random
from threading import Lock
import concurrent.futures
import urllib.request

# our goal is to design multi-threading program that improves the speed of this toy example

class GPU:
    def process(self, node):
        sleep(1)
        val=node.val+1
        print("processed")
        node.val=val
        node.lock.release()
        return val

def process_agent(gpu, node):
    return gpu.process(node)

class Node:
    def __init__(self):
        self.val=0
        self.lock=Lock()

class Tree:
    def __init__(self, gpu_executor):
        # like nodes
        self.gpu_executor=gpu_executor
        self.nodes=[]
        for i in range(10):
            self.nodes.append(Node())
        self.gpu=GPU()
        self.pending_tasks={}

    def run(self):
        epoch=1000
        for i in range(epoch):
            # some computation, but not expensive
            sleep(0.1)
            idx=random.randint(0,9)
            # processing is slow
            # the next iteration might not choose the same index, so it is not necessary to wait for it to finish
            # however, if a node selected is being processed, you must wait for it to finish
            selected_node=self.nodes[idx]
            selected_node.lock.acquire()
            self.pending_tasks[self.gpu_executor.submit(process_agent, self.gpu, selected_node)]=selected_node
            print("submitted")
            # # ret=self.gpu.process(selected_node)
            # selected_node.val=ret
            # print(ret)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu_executor:
        tree=Tree(gpu_executor)
        tree.run()