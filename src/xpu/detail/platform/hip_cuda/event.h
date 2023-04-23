#ifndef XPU_DETAIL_PLATFORM_HIP_CUDA_EVENT_H
#define XPU_DETAIL_PLATFORM_HIP_CUDA_EVENT_H

#include "prelude.h"

namespace xpu::detail {

class event {

public:
    event() {
       [[maybe_unused]] int err = CUHIP(EventCreate)(&m_event);
    }

    ~event() {
        if (m_event) {
            [[maybe_unused]] int err = CUHIP(EventDestroy)(m_event);
        }
    }

    void record(CUHIP(Stream_t) stream) const {
        [[maybe_unused]] int err = CUHIP(EventRecord)(m_event, stream);
    }

    void wait() const {
        [[maybe_unused]] int err = CUHIP(EventSynchronize)(m_event);
    }

    CUHIP(Event_t) handle() const {
        return m_event;
    }

private:
    CUHIP(Event_t) m_event = nullptr;

};

class gpu_timer {

public:
    void start(CUHIP(Stream_t) stream) const {
        m_start.record(stream);
    }

    void stop(CUHIP(Stream_t) stream) const {
        m_stop.record(stream);
    }

    double elapsed() const {
        float ms;
        m_stop.wait();
        [[maybe_unused]] int err = CUHIP(EventElapsedTime)(&ms, m_start.handle(), m_stop.handle());
        return ms;
    }

private:
    event m_start;
    event m_stop;
};

} // namespace xpu::detail

#endif // XPU_DETAIL_PLATFORM_HIP_CUDA_EVENT_H
