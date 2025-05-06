/**
 * @FilePath     : /learning-smorgasbord/C++/crafting interpreter/charpter 14/main.c
 * @Author       : CocoTang1024 1972555958@qq.com
 * @Date         : 2024-12-01 13:08:11
 * @Version      : 0.0.1
 * @LastEditTime : 2024-12-01 13:08:11
 * @Email        : robotym@163.com
 * @Description  : 
**/
#include "common.h"
#include "chunk.h"

int main(int argc, const char* argv[]) {
    Chunk chunk;
    initChunk(&chunk);
    writeChunk(&chunk, OP_RETURN);
    freeChunk(&chunk);
    return 0;
}