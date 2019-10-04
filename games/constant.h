#pragma once

// 自对弈总局数
#define GAME 50000000

// 单局步数限制
#define GAME_LIMIT 800

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
#define EVO_BATCH 1
#endif

#ifdef NDEBUG
#define EVA_SERVER_NUM 3
#else
#define EVA_SERVER_NUM 1
#endif

// 计算推断速度的时间阈值
#define TIME_LIMIT 20

// threshold the only_eat domain
#define THRESHOLD_ONLY_EAT 300

#define LONG_SITUATION_THRESHOLD 3
