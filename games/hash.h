#pragma once

/* wrapper of the fifo of libck
 */
// 这里可能存在内存泄露的问题, 只有单写入多读取模式
extern "C" {
#include <ck_ht.h>
}
#include <pthread.h>

#include <memory>
#include <random>
template <typename KEY, typename VALUE>
class hashtable
{
    const size_t size     = sizeof(VALUE);
    const size_t key_size = sizeof(KEY);
    pthread_mutex_t mtx   = PTHREAD_MUTEX_INITIALIZER;
    ck_ht_t ht;
    // hashtable ck 的辅助用的malloc和free函数.
    static void* ht_malloc(size_t r)
    {
        return malloc(r);
    }

    static void ht_free(void* p, size_t b, bool r)
    {
        (void)b;
        (void)r;
        free(p);
        return;
    }
    struct ck_malloc ck_allocator = {.malloc = ht_malloc, .free = ht_free};

public:
    hashtable()
    {
        if (ck_ht_init(&ht, CK_HT_MODE_BYTESTRING | CK_HT_WORKLOAD_DELETE, NULL,
                       &ck_allocator, 10, 42) == false)
        {
            throw std::runtime_error("ck hash表初始化失败.");
        }
    }
    ~hashtable()
    {
        ck_ht_destroy(&ht);
    }
    // 向hashtable中加入一个元素.
    bool put(KEY key, VALUE* value)
    {
        ck_ht_entry_t entry;
        ck_ht_hash_t h;

        ck_ht_hash_direct(&h, &ht, (uintptr_t)key);
        ck_ht_entry_set_direct(&entry, h, (uintptr_t)key, (uintptr_t)value);
        pthread_mutex_lock(&mtx);
        auto b = ck_ht_put_spmc(&ht, h, &entry);
        pthread_mutex_unlock(&mtx);
        return b;
    }
    bool get(KEY key, VALUE** value)
    {
        ck_ht_entry_t entry;
        ck_ht_hash_t h;

        ck_ht_hash_direct(&h, &ht, (uintptr_t)key);
        ck_ht_entry_key_set_direct(&entry, (uintptr_t)key);
        if (ck_ht_get_spmc(&ht, h, &entry))
        {
            *value = (VALUE*)ck_ht_entry_value(&entry);
            return true;
        }
        else
        {
            return false;
        }
    }
    bool remove(KEY key)
    {
        ck_ht_entry_t entry;
        ck_ht_hash_t h;
        ck_ht_hash_direct(&h, &ht, (uintptr_t)key);
        ck_ht_entry_key_set_direct(&entry, (uintptr_t)key);
        pthread_mutex_lock(&mtx);
        auto r = ck_ht_remove_spmc(&ht, h, &entry);
        pthread_mutex_unlock(&mtx);
        return r;
    }

    bool exists(KEY key)
    {
        VALUE** ptr;
        return get(key, &ptr);
    }
    // hashset 中有多少个元素.
    size_t count()
    {
        return ck_hs_count(ht);
    }
    bool gc()
    {
        static std::random_device rd;
        static std::uniform_int_distribution<uint64_t> dist(0, 1000);
        pthread_mutex_lock(&mtx);
        auto r = ck_hs_gc(ht, 2, dist(rd));
        pthread_mutex_unlock(&mtx);
        return r;
    }
};
