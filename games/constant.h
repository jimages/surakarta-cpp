#pragma once

// 自对弈总局数
#define GAME 50000000

// 单局步数限制
#define GAME_LEN_LIMIT 800

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
#define EVO_BATCH 8
#else
#define EVO_BATCH 4
#endif

// 计算推断速度的时间阈值
#define TIME_LIMIT 20

// threshold the only_eat domain
#define THRESHOLD_ONLY_EAT 300

#define LONG_SITUATION_THRESHOLD 3

#define VIRTUAL_LOSS 1000

#define PB_C_BASE 1599.0f
#define PB_C_INIT 1.25f
#ifdef NDEBUG
#define SIMULATION 800
#else
#define SIMULATION 100
#endif // NDEBUG

// mcts simulation in match mode.
#ifdef NDEBUG
#define SIMULATION_MATCH 20000
#else
#define SIMULATION_MATCH 1500
#endif
