/**
 * @FilePath     : /learning-smorgasbord/C++/crafting interpreter/charpter 14/chunk.h
 * @Author       : CocoTang1024 1972555958@qq.com
 * @Date         : 2024-12-01 13:13:29
 * @Version      : 0.0.1
 * @LastEditTime : 2024-12-01 13:37:21
 * @Email        : robotym@163.com
 * @Description  : 
**/
#ifndef clox_chunk_h
#define clox_chunk_h

#include "common.h"

typedef enum {
    OP_RETURN,
} OpCode;

typedef struct {
    uint8_t* code;
} Chunk;

typedef struct {
    int count;
    int capacity;
    uint8_t* code;
} Chunk;

void initChunk(Chunk* chunk);
void freeChunk(Chunk* chunk);
void writeChunk(Chunk* chunk, uint8_t byte);
#endif