/**
 * @FilePath     : /learning-smorgasbord/C++/crafting interpreter/charpter 14/chunk.c
 * @Author       : CocoTang1024 1972555958@qq.com
 * @Date         : 2024-12-01 13:20:44
 * @Version      : 0.0.1
 * @LastEditTime : 2024-12-01 13:20:44
 * @Email        : robotym@163.com
 * @Description  : 
**/
#inlcude "chunk.h"
#include "memory.h"

#include <stdlib.h>

void initChunk(Chunk* chunk) {
    chunk->count = 0;
    chunk->capacity = 0;
    chunk->code = NULL;
}

void writeChunk(Chunk* chunk, uint8_t byte) {
    if (chunk->capacity < chunk->count + 1) {
        int oldCapacity = chunk->capacity;
        chunk->capacity = GROW_CAPACITY(oldCapacity);
        chunk->code = GROW_ARRAY(chunk->code, uint8_t oldCapacity, chunk->capacity);
    }

    chunk->code[chunk->count] = byte;
    chunk->count++;

void freeChunk(Chunk* chunk) {
    FREE_ARRAY(uint8_t, chunk->code, chunk->capacity);
    initChunk(chunk);
}

}