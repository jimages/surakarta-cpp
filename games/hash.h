#pragma once

/* wrapper of the fifo of libck
 */
// 这里可能存在内存泄露的问题.
#include <ck_ht.h>
#include <random>
#include <memory>
template <typename KEY, typename VALUE>
class hashtable {
    const size_t size = sizeof(VALUE);
    const size_t key_size = sizeof(KEY);
    ck_ht_hash_t ht;
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
    constexpr static struct ck_malloc ck_allocator = {
        .malloc = ht_malloc,
        .free = ht_free
    };

public:
    hashtable()
    {
        if (ck_ht_init(&ht, CK_HT_MODE_BYTESTRING | CK_HT_WORKLOAD_DELETE, ck_allocator, 10, 42) == false) {
            throw std::runtime_error("ck hash表初始化失败.");
        }
    }
    ~hashtable()
    {
        ck_ht_destroy(&ht);
    }
    // 向hashtable中加入一个元素.
    void set(KEY* key, VALUE* value)
    {

        ck_ht_entry_t entry;
        ck_ht_hash_t h;
        ck_ht_hash(&h, &ht, (void*)key, (void*)key_size);
        ck_ht_entry_set(&entry, h, (void*)key, key_size, (void*)value);
        return ck_ht_set_spmc(&ht, h, &entry);
    }
    bool get(KEY* key, VALUE** value)
    {

        ck_ht_entry_t entry;
        ck_ht_hash_t h;
        ck_ht_hash(&h, &ht, (void*)key, (void*)key_size);
        ck_ht_entry_set(&entry, h, (void*)key, key_size, (void*)value);
        if (ck_ht_get_spmc(&ht, h, &entry)) {
            *value = (VALUE*)ck_ht_entry_key(&entry);
            return true;
        } else {
            return false;
        }
    }
    bool remove(KEY* key) {
        ck_ht_entry_t entry;
        ck_ht_hash_t h;
        ck_ht_hash(&h, &ht, (void*)key, (void*)key_size);
        ck_ht_entry_key_set(&entry, h, (void*)key, key_size);
        return ck_ht_remove_spmc(&ht, h, &entry);
    }

    bool exists(KEY* key) {
        VALUE**  ptr;
        return get(key, &ptr);
    }
    // hashset 中有多少个元素.
    size_t count() {
        return ck_hs_count(ht);
    }
    bool gc() {
        static std::random_device rd;
        static std::uniform_int_distribution<uint64_t> dist(0, 1000);
        return ck_hs_gc(ht, 2, dist(rd));
    }
};
