#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

class TpComm {
public:
    static constexpr size_t kRingSlots = 8;
    static constexpr size_t kP2pDecSlotFloats = 65536;
    static constexpr int64_t kP2pPipeElems = 20LL << 20;
    static constexpr int kP2pPipeSlots = 8;
    static constexpr size_t kP2pHeadArgBytes = 16384;
    static constexpr size_t kP2pQsaSlots = 8;
    static constexpr size_t kP2pQsaSelInts = 32768;

    TpComm() = default;
    ~TpComm();
    TpComm(const TpComm&) = delete;
    TpComm& operator=(const TpComm&) = delete;

    bool init(int dev0, int dev1, size_t max_floats, std::string& err);
    void release();
    void reset(uint32_t base = 0);
    void reset_main_flags_async(void* stream0, void* stream1, uint32_t base = 0);
    void mark_flag_dirty() { flag_dirty_ = true; }
    bool consume_flag_dirty() { const bool d = flag_dirty_; flag_dirty_ = false; return d; }
    struct P2pPushTarget {
        float* peer_slot_buf = nullptr;
        volatile uint32_t* peer_flag = nullptr;
        uint32_t* done_ctr = nullptr;
        uint32_t step = 0;
    };
    void allreduce_add(float* local_buf, int n_elems, uint32_t step, int rank, void* stream);
    bool allreduce_add_gr_norm(float* local_buf, float* R_base, const float* inj_base,
                               const float* w_norm, float* xn_out, float* rs_out,
                               float eps, int n_tok, int N, int HC,
                               uint32_t step, int rank, void* stream, bool also_write_bo = false);
    bool p2p_push_target(int n_elems, uint32_t step, int rank, P2pPushTarget& out) const;
    void allreduce_wait_add_bf16(float* local_buf, int n_elems, uint32_t step, int rank, void* stream);
    void allreduce_add_large(float* local_buf, int64_t n_elems, uint32_t& step, int rank, void* stream);
    const uint16_t* p2p_push_bf16_oneshot(const float* local_buf, int64_t n_elems, uint32_t& step, int rank, void* stream);
    uint32_t* p2p_main_flag(int rank) const { return (p2p_dec_ && rank >= 0 && rank < 2) ? d_p2p_flag_[rank] : nullptr; }
    uint32_t* host_main_flag_dev(int rank) const { return (rank >= 0 && rank < 2) ? m_flag_[rank][rank] : nullptr; }
    void* p2p_head_arg(int dst_rank) const { return (p2p_dec_ && dst_rank >= 0 && dst_rank < 2) ? d_p2p_head_arg_[dst_rank] : nullptr; }
    uint32_t* p2p_head_flag(int dst_rank) const { return (p2p_dec_ && dst_rank >= 0 && dst_rank < 2) ? d_p2p_head_flag_[dst_rank] : nullptr; }
    void* p2p_df2_buf(int dst_rank) const { return (p2p_ && p2p_vmm_ && dst_rank >= 0 && dst_rank < 2) ? d_p2p_tx_[dst_rank][0] : nullptr; }
    uint32_t* p2p_df2_flag(int dst_rank) const { return (p2p_ && p2p_vmm_ && dst_rank >= 0 && dst_rank < 2) ? d_p2p_df2_flag_[dst_rank] : nullptr; }
    bool p2p_qsa_ready() const { return p2p_ && p2p_vmm_ && d_p2p_qsa_sel_[0] != nullptr && d_p2p_qsa_flag_[0] != nullptr; }
    void p2p_push_qsa_sel_dec(const int32_t* src, int slot, int dst_off_ints, int n_ints, uint32_t sig_val, int rank, void* stream);
    void p2p_recv_qsa_sel_dec(int32_t* dst, int slot, int src_off_ints, int n_ints, uint32_t wait_val, bool reset_after, int rank, void* stream);
    void p2p_push_qsa_sel_pf(const int32_t* src, int64_t n_ints, int rank, void* stream);
    void p2p_recv_qsa_sel_pf(int32_t* dst, int64_t n_ints, int rank, void* stream);
    bool p2p_gather_logits_rank0(const float* logits0_dev, const float* logits1_dev,
                                 int T, int64_t half0, int64_t half1, int64_t full_vocab,
                                 float* full_logits_dev0, void* stream0, void* stream1);
    bool p2p_swap_expert_blobs(void* slot0_dev, void* slot1_dev, size_t bytes,
                               void* stream0, void* stream1);
    size_t max_floats() const { return max_floats_; }
    size_t slot_floats() const { return slot_floats_; }
    int64_t p2p_pipe_elems() const { return p2p_pipe_elems_; }
    int dev(int r) const { return (r >= 0 && r < 2) ? dev_[r] : 0; }
    bool p2p() const { return p2p_; }
    bool p2p_vmm() const { return p2p_vmm_; }

private:
    bool ensure_pipe(int rank);
    bool ensure_p2p_pipe(int rank);
    void release_pipe();

    int dev_[2] = {0, 1};
    size_t max_floats_ = 0;
    size_t slot_floats_ = 0;
    float* h_buf_[2] = {nullptr, nullptr};
    uint32_t* h_flag_[2] = {nullptr, nullptr};
    float* m_buf_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    uint32_t* m_flag_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};

    // NVLink / P2P device buffers (allocated upfront in init() across both GPUs)
    float* d_dec_buf_[2] = {nullptr, nullptr};
    void* d_p2p_tx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    uint32_t* d_p2p_flag_[2] = {nullptr, nullptr};
    uint32_t* d_done_ctr_[2] = {nullptr, nullptr};
    void* d_p2p_head_arg_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_head_flag_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_logits_flag_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_swap_flag_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_df2_flag_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_qsa_flag_[2] = {nullptr, nullptr};
    int32_t* d_p2p_qsa_sel_[2] = {nullptr, nullptr};
    int32_t* d_p2p_qsa_pf_sel_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_qsa_pf_rdy_[2] = {nullptr, nullptr};
    uint32_t* d_p2p_qsa_pf_ack_[2] = {nullptr, nullptr};
    uint32_t qsa_pf_step_[2][2] = {{0, 0}, {0, 0}};
    uint32_t logits_step_ = 0;
    uint32_t swap_step_ = 0;
    int64_t p2p_pipe_elems_ = kP2pPipeElems;
    bool flag_dirty_ = true;
    // Optional VMM backing (selective peer BAR1 mapping without process-wide cudaDeviceEnablePeerAccess)
    void* vmm_base_[2] = {nullptr, nullptr};
    uint64_t vmm_handle_[2] = {0, 0};
    size_t vmm_bytes_[2] = {0, 0};
    bool p2p_vmm_ = false;

    void* d_tx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    void* d_rx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    void* s_rx_[2] = {nullptr, nullptr};
    void* ev_bar_[2] = {nullptr, nullptr};
    void* ev_done_[2] = {nullptr, nullptr};
    void* ev_rx_[2][2] = {{nullptr, nullptr}, {nullptr, nullptr}};
    void* ev_p2p_rx_[2][kP2pPipeSlots] = {};
    size_t pipe_bytes_ = 0;
    bool pipe_init_[2] = {false, false};
    bool p2p_pipe_init_[2] = {false, false};
    bool p2p_ = false;
    bool p2p_dec_ = false;
};

}  // namespace strata::core
