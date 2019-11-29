from threading import Semaphore, BoundedSemaphore

if __name__ == '__main__':
    sema=Semaphore()
    print("val",sema._value)
    print(1)
    sema.acquire(blocking=False)
    print("val",sema._value)
    print(2)
    sema.acquire(blocking=False)
    print("val",sema._value)
    sema.acquire(blocking=False)
    print("val",sema._value)
    sema.acquire(blocking=False)
    print("val",sema._value)
    print(3)
    sema.acquire()
    print("val",sema._value)
    print(4)