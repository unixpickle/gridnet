class ReLU;
class LeakyReLU;
class SiLU;
class Tanh;

#define APPLY_ACTIVATION(fn, activation)                                        \
    if (activation == "relu") {                                                 \
        fn(ReLU);                                                               \
    } else if (activation == "leaky_relu") {                                    \
        fn(LeakyReLU);                                                          \
    } else if (activation == "silu") {                                          \
        fn(SiLU);                                                               \
    } else if (activation == "tanh") {                                          \
        fn(Tanh);                                                               \
    } else {                                                                    \
        throw std::runtime_error("unknown activation function: " + activation); \
    }

template <typename scalar_t>
__device__ __forceinline__ scalar_t
sigmoid(scalar_t x)
{
    bool negated = x > 0;
    if (negated) {
        x = -x;
    }
    float ex = __expf((float)x);
    scalar_t sig = (scalar_t)(ex / (1.0 + ex));
    if (negated) {
        sig = 1 - sig;
    }
    return sig;
}

template <typename T, typename scalar_t>
class Activation
{
public:
    static const bool implemented = false;

    static __device__ scalar_t forward(scalar_t x)
    {
        return x;
    }
    static __device__ scalar_t backward(scalar_t x)
    {
        return 1.0;
    }
};

template <typename scalar_t>
class Activation<ReLU, scalar_t>
{
public:
    static const bool implemented = true;

    static __device__ __forceinline__ scalar_t forward(scalar_t x)
    {
        if (x < 0) {
            return 0.0;
        } else {
            return x;
        }
    }

    static __device__ __forceinline__ scalar_t backward(scalar_t x)
    {
        if (x < 0) {
            return 0.0;
        } else {
            return 1.0;
        }
    }
};

template <typename scalar_t>
class Activation<LeakyReLU, scalar_t>
{
public:
    static const bool implemented = true;

    static __device__ __forceinline__ scalar_t forward(scalar_t x)
    {
        if (x < 0) {
            return 0.01 * x;
        } else {
            return x;
        }
    }

    static __device__ __forceinline__ scalar_t backward(scalar_t x)
    {
        if (x < 0) {
            return 0.01;
        } else {
            return 1.0;
        }
    }
};

template <typename scalar_t>
class Activation<SiLU, scalar_t>
{
public:
    static const bool implemented = true;

    static __device__ __forceinline__ scalar_t forward(scalar_t x)
    {
        return sigmoid(x) * x;
    }

    static __device__ __forceinline__ scalar_t backward(scalar_t x)
    {
        scalar_t sig = sigmoid(x);
        //   d/dx x*sigmoid(x)
        // = sigmoid(x) + x*sigmoid(x)*(1-sigmoid(x))
        // = sigmoid(x) * (1 + x*(1-sigmoid(x)))
        return sig * (1 + x * (1 - sig));
    }
};

template <typename scalar_t>
class Activation<Tanh, scalar_t>
{
public:
    static const bool implemented = true;

    static __device__ __forceinline__ scalar_t forward(scalar_t x)
    {
        return 2 * sigmoid(2 * x) - 1;
    }

    static __device__ __forceinline__ scalar_t backward(scalar_t x)
    {
        scalar_t s = sigmoid(2 * x);
        return 4 * s * (1 - s);
    }
};