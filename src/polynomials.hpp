#pragma once
#include <type_traits>
#include <cmath>

namespace poly
{
	template<int n, int k>
	struct combination
	{
		static constexpr int64_t value = combination<n - 1, k - 1>::value + combination<n - 1, k>::value;
	};

	template<int n>
	struct combination<n, 0>
	{
		static constexpr int64_t value = 1;
	};

	template<int n>
	struct combination<0, n>
	{
		static constexpr int64_t value = 1;
	};

	template<int n>
	struct combination<n, n>
	{
		static constexpr int64_t value = 1;
	};

	template<>
	struct combination<0, 0>
	{
		static constexpr int value = 1;
	};

	template<int n>
	struct even_odd
	{
		static constexpr int value = -1;
	};
	template<>
	struct even_odd<0>
	{
		static constexpr int value = 1;
	};

	template<int _Order, class _Type, int n = 0>
	struct shiftedLegendre
	{
		inline static _Type at(_Type x)
		{
			return shiftedLegendre<_Order, _Type, n + 1>::at(x) * x + combination<_Order, n>::value * combination<_Order + n, n>::value * even_odd<(_Order + n) % 2>::value;
		}
	};

	template<int _Order, class _Type>
	struct shiftedLegendre<_Order, _Type, _Order>
	{
		inline static _Type at(_Type x)
		{
			return combination<_Order + _Order, _Order>::value;
		}
	};

	template<class _Type>
	struct shiftedLegendre<0, _Type, 0>
	{
		inline static _Type at(_Type x)
		{
			return 1;
		}
	};

	template<int _Order, class _Type> inline _Type shiftedLegendreFunc(_Type x)
	{
		return shiftedLegendre<_Order, _Type, 0>::at(x);
	}


	template<class _Type> _Type slpGet(size_t order, _Type x)
	{
		switch (order)
		{
		case 0: return shiftedLegendreFunc<0>(x);
		case 1: return shiftedLegendreFunc<1>(x);
		case 2: return shiftedLegendreFunc<2>(x);
		case 3: return shiftedLegendreFunc<3>(x);
		case 4: return shiftedLegendreFunc<4>(x);
		case 5: return shiftedLegendreFunc<5>(x);
		case 6: return shiftedLegendreFunc<6>(x);
		case 7: return shiftedLegendreFunc<7>(x);
		case 8: return shiftedLegendreFunc<8>(x);
		case 9: return shiftedLegendreFunc<9>(x);
		case 10: return shiftedLegendreFunc<10>(x);
		case 11: return shiftedLegendreFunc<11>(x);
		case 12: return shiftedLegendreFunc<12>(x);
		case 13: return shiftedLegendreFunc<13>(x);
		case 14: return shiftedLegendreFunc<14>(x);
		case 15: return shiftedLegendreFunc<15>(x);
		}
		return _Type{};
	}

	inline size_t partialProductDown(size_t n, size_t k)
	{
		size_t ret = 1;
		for (size_t i = 0; i < k; ++i) ret *= n--;
		return ret;
	}

	inline int slpGetCoef(size_t n, size_t k)
	{
		return ((n + k) & 1 ? -1 : 1) * (int)(partialProductDown(n, k) / partialProductDown(k, k) * partialProductDown(n + k, k) / partialProductDown(k, k));
	}

	template<class _Type, int _Order> struct chebyshevU;

	template<class _Type, int _Order>
	struct chebyshevT
	{
		inline static _Type get(_Type x)
		{
			if(_Order % 2) return 2 * chebyshevT<_Type, _Order / 2>::get(x) * chebyshevT<_Type, _Order / 2 + 1>::get(x) - x;
			else return 2 * std::pow(chebyshevT<_Type, _Order / 2>::get(x), 2) - 1;
		}

		inline static _Type derived(_Type x)
		{
			return _Order * chebyshevU<_Type, _Order - 1>::get(x);
		}
	};

	template<class _Type>
	struct chebyshevT<_Type, 0>
	{
		inline static _Type get(_Type x)
		{
			return 1;
		}

		inline static _Type derived(_Type x)
		{
			return 0;
		}
	};

	template<class _Type>
	struct chebyshevT<_Type, 1>
	{
		inline static _Type get(_Type x)
		{
			return x;
		}

		inline static _Type derived(_Type x)
		{
			return 1;
		}
	};

	template<int _Order, class _Type> inline _Type chebyshevTf(_Type x)
	{
		return chebyshevT<_Type, _Order>::get(x);
	}

	template<int _Order, class _Type> inline _Type chebyshevTg(_Type x)
	{
		return chebyshevT<_Type, _Order>::derived(x);
	}

	template<class _Type> _Type chebyshevTGet(size_t order, _Type x)
	{
		switch (order)
		{
		case 0: return chebyshevTf<0>(x);
		case 1: return chebyshevTf<1>(x);
		case 2: return chebyshevTf<2>(x);
		case 3: return chebyshevTf<3>(x);
		case 4: return chebyshevTf<4>(x);
		case 5: return chebyshevTf<5>(x);
		case 6: return chebyshevTf<6>(x);
		case 7: return chebyshevTf<7>(x);
		case 8: return chebyshevTf<8>(x);
		case 9: return chebyshevTf<9>(x);
		case 10: return chebyshevTf<10>(x);
		case 11: return chebyshevTf<11>(x);
		case 12: return chebyshevTf<12>(x);
		case 13: return chebyshevTf<13>(x);
		case 14: return chebyshevTf<14>(x);
		case 15: return chebyshevTf<15>(x);
		}
		if(order % 2) return 2 * chebyshevTGet(order / 2, x) * chebyshevTGet(order / 2 + 1, x) - x;
		return 2 * pow(chebyshevTGet(order / 2, x), 2) - 1;
	}

	template<class _Type> _Type chebyshevTDerived(size_t order, _Type x)
	{
		switch (order)
		{
		case 0: return chebyshevTg<0>(x);
		case 1: return chebyshevTg<1>(x);
		case 2: return chebyshevTg<2>(x);
		case 3: return chebyshevTg<3>(x);
		case 4: return chebyshevTg<4>(x);
		case 5: return chebyshevTg<5>(x);
		case 6: return chebyshevTg<6>(x);
		case 7: return chebyshevTg<7>(x);
		case 8: return chebyshevTg<8>(x);
		case 9: return chebyshevTg<9>(x);
		case 10: return chebyshevTg<10>(x);
		case 11: return chebyshevTg<11>(x);
		case 12: return chebyshevTg<12>(x);
		case 13: return chebyshevTg<13>(x);
		case 14: return chebyshevTg<14>(x);
		case 15: return chebyshevTg<15>(x);
		}
		return _Type{};
	}

	template<class _Type, int _Order>
	struct chebyshevU
	{
		inline static _Type get(_Type x)
		{
			return chebyshevU<_Type, _Order - 2>::get(x) * (4 * pow(x, 2) - 2) - chebyshevU<_Type, _Order - 4>::get(x);
		}
	};

	template<class _Type>
	struct chebyshevU<_Type, 0>
	{
		inline static _Type get(_Type x)
		{
			return 1;
		}
	};

	template<class _Type>
	struct chebyshevU<_Type, 1>
	{
		inline static _Type get(_Type x)
		{
			return 2 * x;
		}
	};

	template<class _Type>
	struct chebyshevU<_Type, 2>
	{
		inline static _Type get(_Type x)
		{
			return 4 * pow(x, 2) - 1;
		}
	};

	template<class _Type>
	struct chebyshevU<_Type, 3>
	{
		inline static _Type get(_Type x)
		{
			return 8 * pow(x, 3) - 4 * x;
		}
	};

	template<int _Order, class _Type> inline _Type chebyshevUf(_Type x)
	{
		return chebyshevU<_Type, _Order>::get(x);
	}
}