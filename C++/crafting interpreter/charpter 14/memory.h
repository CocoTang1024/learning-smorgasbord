/**
 * @FilePath     : /learning-smorgasbord/C++/crafting interpreter/charpter 14/memory.h
 * @Author       : CocoTang1024 1972555958@qq.com
 * @Date         : 2024-12-01 13:27:32
 * @Version      : 0.0.1
 * @LastEditTime : 2024-12-01 13:38:47
 * @Email        : robotym@163.com
 * @Description  : 
**/
#ifndef clox_memory_h
#define clox_memory_h

#include "common.h"
#define GROW_CAPACITY(capacity) \
    ((capacity) < 8 ? 8 : (capacity) * 2)

#define GROW_ARRAY(type, pointer, oldCount, newCount) \
    (type*)reallocate(pointer, sizeof(type) * (oldCount), \
    sizeof(type) * (newCount))

#define FREE_ARRAY(type, pointer, oldCount) \
    reallocate(pointer, sizeof(type) * (oldCount), 0)
    
void* reallocate(void * pointer, size_t oldSize, size_t newSize);

#endif