#ifndef CONSTANT_H
#define CONSTANT_H

// 自对弈总局数
#define GAME 50000000

// 单局步数限制
#define GAME_LIMIT 300

// 训练batch
#ifdef NDEBUG
#define SAMPLE_SIZE 4096
#else
#define SAMPLE_SIZE 1024
#endif

// 训练数据窗口数量
#define GAME_DATA_LIMIT 1000000

// evaluation的batch数量
#ifdef NDEBUG
#define EVO_BATCH 4
#else
#define EVO_BATCH 3
#endif

#ifdef NDEBUG
#define EVA_SERVER_NUM 3
#else
#define EVA_SERVER_NUM 2
#endif

// 计算推断速度的时间阈值
#define TIME_LIMIT 5

#endif // CONSTANT_H