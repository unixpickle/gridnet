class ReLU;
class SiLU;

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