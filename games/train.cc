/*
 * 自对弈训练程序
 */
#include <omp.h>
#include <pthread.h>
#include <spdlog/cfg/env.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <torch/torch.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cstdlib>
#include <deque>
#include <functional>
#include <future>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "concurrentqueue.h"
#include "constant.h"
#include "hash.h"
#include "helper.h"
#include "mcts.h"
#include "policy_value_model.h"
#include "surakarta.h"

using MCTS::Node;
using torch::Tensor;
// 训练数据集的队列
moodycamel::ConcurrentQueue<std::array<Tensor, 3>> train_dataset(100);

// 前向计算数据集的队列
moodycamel::ConcurrentQueue<std::pair<decltype(omp_get_thread_num()), Tensor>>
    evo_queue(50 * decltype(evo_queue)::BLOCK_SIZE);
pthread_cond_t evo_cond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t evo_mtx = PTHREAD_MUTEX_INITIALIZER;
// 向自对弈线程返回计算结果
hashtable<decltype(omp_get_thread_num()), std::pair<Tensor, Tensor>> evo_ht;

// 当训练线程加载完毕模型之后,前向计算线程从复制模型
pthread_cond_t model_cond      = PTHREAD_COND_INITIALIZER;
pthread_mutex_t model_cond_mtx = PTHREAD_MUTEX_INITIALIZER;

// 用于训练的Network,设置为全局变量
PolicyValueNet gNetwork;
pthread_mutex_t model_mtx = PTHREAD_MUTEX_INITIALIZER;

// 用于标记模型是否更新的量,注意使用的是原子量
std::atomic_ullong model_version = 0;

// 多线程中用于计算前向计算的次数的值.
std::atomic_ullong evo_counter = 0;

Tensor get_statistc(std::shared_ptr<Node<SurakartaState>> node)
{
    Tensor prob = torch::zeros({SURAKARTA_ACTION}, torch::kFloat);
    float total = std::accumulate(
        (node->children).begin(), (node->children).end(), 0.0,
        [](const float l, const Node<SurakartaState>::move_node_tuple& r) {
            return l + r.second->visits;
        });
    for (auto i = (node->children).begin(); i != (node->children).end(); ++i)
    {
        prob[move2index(i->first)] = i->second->visits / total;
    }
    prob.masked_fill_(torch::isnan(prob), 0);
    return prob.unsqueeze_(0);
}

std::tuple<Tensor, Tensor, Tensor> sample(const Tensor& board,
                                          const Tensor& mcts,
                                          const Tensor& value)
{
    std::random_device rd;
    std::mt19937 g(rd());

    // generate the sample indexs.
    auto idx =
        torch::randperm(board.size(0), torch::TensorOptions(torch::kInt64))
            .slice(0, 0, 1024, 1);

    return std::make_tuple(board.index_select(0, idx),
                           mcts.index_select(0, idx),
                           value.index_select(0, idx));
}

void* trainer(void* param)
{
    SPDLOG_INFO("单步的模拟次数为: {}", SIMULATION);
    SPDLOG_INFO("单次前向计算的数量为: {}", EVO_BATCH);
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::info);

    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(
        "logs/train.log", true);
    file_sink->set_level(spdlog::level::info);

    spdlog::logger logger("multi_sink", {console_sink, file_sink});
    // 加载model阶段,获取model的锁
    pthread_mutex_lock(&model_mtx);

    // 初始化相关的数据
    uint64_t game = 1;
    Tensor board  = torch::empty({0});
    Tensor mcts   = torch::empty({0});
    Tensor value  = torch::empty({0});
    std::string model_buf;

    // load the pt if exists
    if (exists("board.pt"))
        torch::load(board, "board.pt");
    if (exists("mcts.pt"))
        torch::load(mcts, "mcts.pt");
    if (exists("value.pt"))
        torch::load(value, "value.pt");
    std::stringstream s;
    s << gNetwork.model;
    // model加载完毕,让evolution开始复制现有的model.
    spdlog::info(s.str());

    if (exists("value_policy.pt"))
        gNetwork.load_model("value_policy.pt");
    if (exists("optimizer.pt"))
        torch::load(*(gNetwork.optimizer), "optimizer.pt");

    if (torch::cuda::is_available())
    {
        gNetwork.model->to(torch::kCPU);
    }
    if (torch::cuda::is_available())
    {
        gNetwork.model->to(torch::kCUDA);
    }

    SPDLOG_INFO("模型加载完毕.");
    // 加载model完毕,施放锁,通知前向计算服务器拷贝模型
    model_version = 1ull;
    pthread_mutex_unlock(&model_mtx);
    pthread_cond_signal(&model_cond);

    std::array<Tensor, 3> d;

    auto last_speed_chk_tim = omp_get_wtime();

    last_speed_chk_tim = omp_get_wtime();
    while (true)
    {
        // 睡眠一秒钟,这样的话不用一直死循环.
        sleep(1);

        // 统计前向计算的速度
        if (last_speed_chk_tim + 10 < omp_get_wtime())
        {
            std::printf("\rspeed:%f",
                        evo_counter / (omp_get_wtime() - last_speed_chk_tim));
            std::fflush(stdout);
            evo_counter        = 0;
            last_speed_chk_tim = omp_get_wtime();
        }
        if (train_dataset.try_dequeue(d))
        {
            // get board, probability, value
            Tensor b, p, v;
            b = d.at(0);
            p = d.at(1).reshape({-1, 6, 6, 6, 6});
            v = d.at(2);
            // 在这里进行一次数据的扩充,镜像扩充.
            board = torch::cat(
                {b, b.flip({2, 3}), b.flip({2}), b.flip({3}), board});
            mcts =
                torch::cat({p.reshape({-1, 6 * 6 * 6 * 6}),
                            p.flip({1, 2, 3, 4})
                                .reshape({-1, 6 * 6 * 6 * 6})
                                .reshape({-1, 6 * 6 * 6 * 6}),
                            p.flip({1, 3}).reshape({-1, 6 * 6 * 6 * 6}),
                            p.flip({2, 4}).reshape({-1, 6 * 6 * 6 * 6}), mcts});
            value = torch::cat({v, v, v, v, value});
            std::cout << '\n';
            spdlog::info("积累的局数: {} 数据集: {} 当局步数: {} ", game,
                         board.size(0), v.size(0));

            if (board.size(0) >= SAMPLE_SIZE)
            {
                // 等样本数量超过限制的时候，去掉头部的数据。
                unsigned long size = board.size(0);
                if (size > GAME_DATA_LIMIT)
                {
                    board = board.narrow(0, size - GAME_DATA_LIMIT - 1,
                                         GAME_DATA_LIMIT);
                    mcts  = mcts.narrow(0, size - GAME_DATA_LIMIT - 1,
                                       GAME_DATA_LIMIT);
                    value = value.narrow(0, size - GAME_DATA_LIMIT - 1,
                                         GAME_DATA_LIMIT);
                }
                // 提取样本
                Tensor loss, entropy;
                std::tie(b, p, v) = sample(board, mcts, value);

                SPDLOG_DEBUG("准备进行一次训练,因此锁住全局模型");
                // 要进行反向传播,所以锁住模型
                pthread_mutex_lock(&model_mtx);
                for (auto i = 0; i <= 5; ++i)
                {
                    std::tie(loss, entropy) = gNetwork.train_step(b, p, v);
                    logger.info(
                        "game: {} loss: {} entropy: {} train "
                        "dataset size: {} model version: {} step version: {}",
                        game, loss.item<float>(), entropy.item<float>(),
                        board.size(0), model_version, i);
                }
                pthread_mutex_unlock(&model_mtx);

                model_version++;
            }

            // saving the model
            if (game % 100 == 0)
            {
                spdlog::info("保存网络, 当前对局数:{}", game);

                pthread_mutex_lock(&model_mtx);

                gNetwork.save_model("value_policy.pt");
                torch::save(board, "board.pt");
                torch::save(mcts, "mcts.pt");
                torch::save(value, "value.pt");
                torch::save(*(gNetwork.optimizer), "optimizer.pt");

                pthread_mutex_unlock(&model_mtx);
            }
            ++game;
        }
    }
    spdlog::error("trainer 线程异常结束");
    return NULL;
}
void* evoluter(void* params)
{
    // 先睡一秒钟,防止出现evoluter比trainer先进入临界区.
    sleep(1);
#ifndef NDEBUG
    omp_set_num_threads(2);
#else
    omp_set_num_threads(4);
#endif
#pragma omp parallel
    {
        moodycamel::ConsumerToken ctok(evo_queue);
        auto id = omp_get_thread_num();
        spdlog::info("前向计算线程{}启动!", id);
        std::deque<int> sender_deque;
        uint64_t evo_model_ver;

        // When run on cpu we should initizlize the model on cpu.
        // So the we assign 1 to totoal_device to bypass the divide_by_zero
        // errors. todo: 处理好自主选择GPU芯片的功能.
        int total_device = torch::cuda::device_count();
        if (!torch::cuda::is_available())
        {
            total_device = 1;
        }

        // 开始拷贝model数据,用于前向计算
        // 直接从主线程复制一个network.
        pthread_mutex_lock(&model_mtx);
        auto dev_idx         = (total_device - 1) - id % total_device;
        auto gNetwork_device = gNetwork.model->parameters()[0].device();
        spdlog::info("evoluter{}开始复制model到device:{}", id, dev_idx);
        PolicyValueNet net(dev_idx);
        net.model = std::dynamic_pointer_cast<NetImpl>(
            gNetwork.model->clone(net.device));
        assert(gNetwork_device == gNetwork.model->parameters()[0].device());

        evo_model_ver = model_version;
        pthread_mutex_unlock(&model_mtx);

        Tensor evolution_batch = torch::empty({0});

#pragma omp barrier
        std::array<std::pair<decltype(omp_get_thread_num()), Tensor>, EVO_BATCH>
            p;
        auto total_tim         = .0;
        double inference_tim   = .0;
        double try_dequeue_tim = .0;
        while (true)
        {
            // 获取自对弈线程发送的棋局数据
            auto tim = omp_get_wtime();
            size_t s = evo_queue.try_dequeue_bulk(ctok, p.begin(), (int)EVO_BATCH * 0.666666);
            if (s)
            {
                for (size_t i = 0; i < s; ++i)
                {
                    sender_deque.emplace_back(p[i].first);
                    evolution_batch =
                        torch::cat({evolution_batch, p[i].second});
                }
            }
            try_dequeue_tim += omp_get_wtime() - tim;

            // 拿到了对于的数据,就开始进行前向计算
            if (sender_deque.size() >= EVO_BATCH)
            {
                Tensor policy, value;
                // 拷贝一份evolution中的数据,防止在多线程计算的时候出现竞争条件.
                auto batch =
                    evolution_batch
                        .index({torch::indexing::Slice({0, 0 + EVO_BATCH})})
                        .clone();
                evolution_batch = evolution_batch.index({torch::indexing::Slice(
                    0 + EVO_BATCH, torch::indexing::None)});
                auto tim        = omp_get_wtime();
                std::tie(policy, value) = net.policy_value(batch);
                inference_tim += omp_get_wtime() - tim;

                assert(policy.size(0) == (long long)batch.size(0));
                assert(sender_deque.size() == (size_t)batch.size(0));

                // 将前向计算后的结果送回自对弈线程
                for (long long ind = 0; ind != batch.size(0); ++ind)
                {
                    // 使用ck_hs, 这里又分配了堆上内存.
                    auto pair = new std::pair(policy[ind], value[ind]);
                    auto key  = sender_deque.front();
                    sender_deque.pop_front();
                    if (evo_ht.put(key, pair) == false)
                    {
                        SPDLOG_CRITICAL("计算:发送给{}失败", key);
                    }
                    evo_counter++;
                }
                // 设置完毕之后,发送一个广播,唤醒所有的等待的线程.
                pthread_cond_broadcast(&evo_cond);
            }
            // 在这里我们设置一起提醒以避免死锁
            pthread_cond_broadcast(&evo_cond);

            total_tim += omp_get_wtime() - tim;
            // 检测距离上一次是否有更新过模型
            if (evo_model_ver != model_version)
            {
                if (pthread_mutex_trylock(&model_mtx) == 0)
                {
                    spdlog::info("准备同步模型");
                    spdlog::debug("推断总共时间 {} 其中推断时间为 {} 占比为 {} "
                                  "dequeue {} 占比为 {}",
                                  total_tim, inference_tim,
                                  inference_tim / total_tim, try_dequeue_tim,
                                  try_dequeue_tim / total_tim);
                    net.model     = Net(std::dynamic_pointer_cast<NetImpl>(
                        gNetwork.model->clone(net.device)));
                    evo_model_ver = model_version;
                    pthread_mutex_unlock(&model_mtx);
                }
            }
        }
    }
    spdlog::error("evolution 线程异常结束");
    return NULL;
}

/* 自对弈的,对弈线程实现
 */
void* worker(void* params)
{
    SPDLOG_INFO("开始自对弈训练,单局步数限制为{}, 单步模拟次数为{}",
                GAME_LEN_LIMIT, SIMULATION);
    // 调试状态下,只开启一个线程
#ifndef NDEBUG
    omp_set_num_threads(8);
#endif
    spdlog::info("自对弈线程数:{}", omp_get_max_threads());
#pragma omp parallel
    {
        moodycamel::ProducerToken ptok(evo_queue);
        while (true)
        {
            // 用于计算空闲时间
            auto start_tm  = omp_get_wtime();
            double busy_tm = .0;

            SPDLOG_INFO("开始一局自对弈");
            Tensor b = torch::zeros({0});
            Tensor p = torch::zeros({0});

            // OK let's play game!
            size_t count = 0;
            SurakartaState game;
            auto root =
                std::make_shared<Node<SurakartaState>>(game.player_to_move);
            auto player = torch::zeros({0}, torch::TensorOptions(torch::kInt));
            // 如果棋局还没有结束并且一方还可以下棋,并且局数大于限制的时候
            while (!game.terminal() && game.has_moves() &&
                   count < GAME_LEN_LIMIT)
            {
                auto board               = game.tensor();
                unsigned int equal_count = 0;

                // 多线程内的消息传递
                auto move = run_mcts(
                    root, game, count / 2,
                    [&busy_tm, &ptok](Tensor t) -> std::pair<Tensor, Tensor> {
                        auto id        = omp_get_thread_num() + 1;
                        auto data      = std::make_pair(id, t);
                        auto start_tim = omp_get_wtime();
                        evo_queue.enqueue(ptok, std::move(data));
                        // 用于存储前向计算结果的指针
                        std::pair<Tensor, Tensor>* p;

                        while (pthread_cond_wait(&evo_cond, &evo_mtx) == 0)
                        {
                            // 前向计算每完成一次计算,就唤醒一次子线程,让他们自己去找有没有自己的id,如果没有就再等等
                            if (evo_ht.get(id, &p))
                            {
                                evo_ht.remove(id);
                                auto d = *p;
                                // ht中的数据是堆上的内存,所以我们应该清空掉这里面的数据
                                delete p;
                                // 计算mcts和前向计算的时间
                                busy_tm += omp_get_wtime() - start_tim;
                                return {d.first, d.second};
                            }
                            else
                            {
                                // 如果没有,那么就等下一轮吧.
                                continue;
                            }
                            // this code will not be run,
                            // make compiler happy
                            return {Tensor(), Tensor()};
                        }
                        SPDLOG_CRITICAL("异常退出了worker");
                    });
                // for long situation.
                for (int i = b.size(0) - 2; i >= 0; i -= 2)
                {
                    if (board[0]
                            .slice(0, 0, 2)
                            .to(torch::kBool)
                            .equal(b[i].slice(0, 0, 2).to(torch::kBool)))
                    {
                        equal_count++;
                    }
                    if (equal_count >= LONG_SITUATION_THRESHOLD)
                    {
                        std::cout << '\n';
                        SPDLOG_WARN("发现了一个循环局面,提前结束这一局");
                        goto out;
                    }
                }
            out:
                b      = at::cat({b, board}, 0);
                p      = at::cat({p, get_statistc(root)}, 0);
                player = torch::cat(
                    {player,
                     torch::from_blob(&game.player_to_move, 1,
                                      torch::TensorOptions(torch::kInt))});

                // 如果如果找到了重复出现的局面,则直接跳出这一局,直接结算结果
                if (equal_count >= LONG_SITUATION_THRESHOLD)
                    goto finish;

                // if in long situation. exit.
                game.do_move(move);
                root = root->get_child(move);
                ++count;
            }
        finish:
            // play 1 or 2, 0 for draw
            int winner = game.get_winner();
            int size   = b.size(0);
            auto v     = torch::zeros({size}, torch::kFloat);
            auto win_result =
                torch::zeros_like(player, torch::TensorOptions(torch::kFloat));
            win_result.masked_fill_((player == winner), 1.0F);
            win_result.masked_fill_((player != winner), -1.0F);

            // 这一局完成后,将比赛的结果发送给训练线程,收集数据
            spdlog::info("自对弈结束一局,该局步数:{}, 胜利方:{}", size, winner);
            auto total_tm = omp_get_wtime() - start_tm;
            spdlog::debug(
                "该局的完成的总时间为: {} 等待推断结果时间: {} 占比为: {}",
                total_tm, busy_tm, busy_tm / total_tm);
            auto data =
                std::array<Tensor, 3>{b, p, win_result.to(torch::kFloat)};
            train_dataset.enqueue(std::move(data));
        }
    }
    spdlog::error("worder 线程异常结束");
    return NULL;
}

int main(int argc, char* argv[])
{
    spdlog::cfg::load_env_levels();
    spdlog::set_pattern(
        "[%H:%M:%S %z] [%n] [%^%l%$] [process %P] [thread %t] %v");
    SPDLOG_INFO("苏拉卡尔塔棋AlphaZero by Zachary Wang.");

    pthread_t t[3];

    SPDLOG_INFO("创建训练线程");
    pthread_create(&t[0], NULL, trainer, NULL);

    SPDLOG_INFO("创建前向计算线程");
    pthread_create(&t[1], NULL, evoluter, NULL);

    SPDLOG_INFO("创建了运算服务器");
    pthread_create(&t[2], NULL, worker, NULL);

    SPDLOG_INFO("开始运行");
    for (auto i : t)
    {
        pthread_join(i, NULL);
    }
    SPDLOG_INFO("结束运行");
    return 0;
}
