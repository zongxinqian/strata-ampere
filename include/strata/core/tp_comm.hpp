#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

class TpComm {
public:
    static constexpr size_t kRingSlots = 8;

    TpComm() = default;
    ~TpComm();
    TpComm(const TpComm&) = delete;
    TpComm& operator=(const TpComm&) = delete;

    bool init(int dev0, int dev1, size_t max_floats, std::string& err);
    void release();
    void reset(uint32_t base = 0);
    void allreduce_add(float* local_buf, int n_elems, uint32_t step, int rank, void* stream);
    void allreduce_add_large(float* local_buf, int64_t n_elems, uint32_t& step, int rank, void* stream);
    size_t max_floats() const { return max_floats_; }
    size_t slot_floats() const { return slot_floats_; }

private:
    bool ensure_pipe(int rank);
    void release_pipe();

    int dev_[2] = {0, 1};
    size_t max_floats_ = 0;
    size_t slot_floats_ = 0;
    float* h_buf_[2] = {nullptr, nullptr};
    uint32_t* h_flag_[2] = {nullptr, nullptr};
    float* m_buf_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    uint32_t* m_flag_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};

    void* d_tx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    void* d_rx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    void* s_rx_[2] = {nullptr, nullptr};
    void* ev_bar_[2] = {nullptr, nullptr};
    void* ev_done_[2] = {nullptr, nullptr};
    void* ev_rx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    size_t pipe_bytes_ = 0;
    bool pipe_init_[2] = {false, false};
};

}  // namespace strata::core
