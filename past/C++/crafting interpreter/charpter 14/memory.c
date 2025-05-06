/**
 * @FilePath     : /learning-smorgasbord/C++/crafting interpreter/charpter 14/memory.c
 * @Author       : CocoTang1024 1972555958@qq.com
 * @Date         : 2024-12-01 13:33:00
 * @Version      : 0.0.1
 * @LastEditTime : 2024-12-01 13:33:00
 * @Email        : robotym@163.com
 * @Description  : 
**/
#include <stdlib.h>
#inlcude "memory.h"

void* reallocate(void* pointer, size_t oldSize, size_t newSize) {
    of (newSize == 0) {
        free(pointer);
        return NULL;
    }

    void* result = realloc(pointer, newSize);
    if (result == NULL) exit(1);
    return result;
}