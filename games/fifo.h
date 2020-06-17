#pragma once

/* wrapper of the fifo of libck
 */
// 这里可能存在内存泄露的问题.
extern "C" {
#include <ck_fifo.h>
}
#include <memory>
template <typename T>
class fifo {
    typedef T Value;
    const size_t size = sizeof(Value);
    ck_fifo_mpmc f;
    static std::allocator<ck_fifo_mpmc_entry> stub_allocator;
    static std::allocator<T> value_allocator;

public:
    fifo()
    {
        ck_fifo_mpmc_init(&f, reinterpret_cast<void*>(stub_allocator.allocate(1)));
    }
    ~fifo()
    {
        ck_fifo_mpmc_entry* garbage;
        ck_fifo_mpmc_deinit(&f, &garbage);
        while (NULL != garbage) {
            auto next = garbage->next.pointer;
            stub_allocator.deallocate(garbage, 1);
            garbage = next;
        }
    }
    void enqueue(T* value)
    {
        ck_fifo_mpmc_enqueue(&f, stub_allocator.allocate(1), reinterpret_cast<void*>(value));
    }
    // 返回True的时候,表示获取成功,否则表示获取失败.
    bool dequeue(T** value)
    {
        ck_fifo_mpmc_entry* entry;
        if (ck_fifo_mpmc_dequeue(&f, reinterpret_cast<void*>(value), &entry) == false) {
            return false;
        } else {
            stub_allocator.deallocate(entry, 1);
            return true;
        }
    }
};
