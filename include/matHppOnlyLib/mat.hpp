/* Change memLayout to fit avx2 kernel
 * and write some code that satisfy the compiler's preference.
 * Anyway, the reality only talks about how strong you are, so
 * any optimization is impossible to beat a true powerful hardware.
 */

/* x86_64 avx2 omp ONLY */
#pragma once

#include <iostream>
#include <random>
#include <unordered_map>
#include <omp.h>
#include <cstring>
#include <immintrin.h>

/* Nowadays, after 2020, 16 threads is common...
* And in this case, MAT_THREAD_NUMS wont corrupt omp's
* parameters when there is another lib uses omp but
* the thread is set differently.
*/
#ifndef MAT_THREAD_NUMS
#define MAT_THREAD_NUMS 16
#endif

#define BROADCASTANDFMADDSF(B) a0 = _mm256_broadcast_ss(leftBasePtr);     \
						       a1 = _mm256_broadcast_ss(leftBasePtr + 1); \
						       a2 = _mm256_broadcast_ss(leftBasePtr + 2); \
						       a3 = _mm256_broadcast_ss(leftBasePtr + 3); \
						       a4 = _mm256_broadcast_ss(leftBasePtr + 4); \
						       a5 = _mm256_broadcast_ss(leftBasePtr + 5); \
						       a6 = _mm256_broadcast_ss(leftBasePtr + 6); \
						       a7 = _mm256_broadcast_ss(leftBasePtr + 7); \
						       c0 = _mm256_fmadd_ps(a0, B, c0);	   	      \
						       c1 = _mm256_fmadd_ps(a1, B, c1);	   	      \
						       c2 = _mm256_fmadd_ps(a2, B, c2);	   	      \
						       c3 = _mm256_fmadd_ps(a3, B, c3);	   	      \
						       c4 = _mm256_fmadd_ps(a4, B, c4);	   	      \
						       c5 = _mm256_fmadd_ps(a5, B, c5);	   	      \
						       c6 = _mm256_fmadd_ps(a6, B, c6);	   	      \
						       c7 = _mm256_fmadd_ps(a7, B, c7);	   	      \
                               leftBasePtr += NumTypeSize<NumType>::L1Y;

#define BROADCASTANDFMADDDF(B) a0 = _mm256_broadcast_sd(leftBasePtr);     \
						       a1 = _mm256_broadcast_sd(leftBasePtr + 1); \
						       a2 = _mm256_broadcast_sd(leftBasePtr + 2); \
						       a3 = _mm256_broadcast_sd(leftBasePtr + 3); \
						       a4 = _mm256_broadcast_sd(leftBasePtr + 4); \
						       a5 = _mm256_broadcast_sd(leftBasePtr + 5); \
						       a6 = _mm256_broadcast_sd(leftBasePtr + 6); \
						       a7 = _mm256_broadcast_sd(leftBasePtr + 7); \
						       c0 = _mm256_fmadd_pd(a0, B, c0);	   	      \
						       c1 = _mm256_fmadd_pd(a1, B, c1);	   	      \
						       c2 = _mm256_fmadd_pd(a2, B, c2);	   	      \
						       c3 = _mm256_fmadd_pd(a3, B, c3);	   	      \
						       c4 = _mm256_fmadd_pd(a4, B, c4);	   	      \
						       c5 = _mm256_fmadd_pd(a5, B, c5);	   	      \
						       c6 = _mm256_fmadd_pd(a6, B, c6);	   	      \
						       c7 = _mm256_fmadd_pd(a7, B, c7);	   	      \
                               leftBasePtr += NumTypeSize<NumType>::L1Y;

/* vec 不分行向量还是列向量, 要区分提升到 mat, 默认是列向量 */
template <typename NumType>
class vec;

/* 默认 ROW-MAJOR */
template <typename NumType>
class mat;

/* 矩阵乘法缓存参数 */
template <typename InNumType>
struct NumTypeSize;

/* 矩阵乘法缓存参数 float */
template <>
struct NumTypeSize<float>;

/* 矩阵乘法缓存参数 double */
template<>
struct NumTypeSize<double>;

/* mat 类型提取 */
template <typename U>
struct MatNumType;

/* mat 提取 NumType */
template <typename T>
struct MatNumType<mat<T>>;

/*****************************************************************/

template <typename InNumType>
struct NumTypeSize {
	static constexpr size_t L3Y = 768; // 768
	static constexpr size_t L3X = 512; // 512
	static constexpr size_t L3Z = 768; // 768
	static constexpr size_t LEFTSIZEL3 = L3Y * L3X;
	static constexpr size_t RIGHTSIZEL3 = L3X * L3Z;

	static constexpr size_t L2Y = 768; // 768
	static constexpr size_t L2X = 256; // 256
	static constexpr size_t L2Z = 768; // 768
	static constexpr size_t LEFTSIZEL2 = L2Y * L2X;
	static constexpr size_t RIGHTSIZEL2 = L2X * L2Z;

	static constexpr size_t L1Y = 8; // 8
	static constexpr size_t L1X = 256; // 256
	static constexpr size_t L1Z = 8; // 8
};

template<>
struct NumTypeSize<float> {

#if MAT_THREAD_NUMS == 1
	static constexpr size_t L3Y = 768; // 768
	static constexpr size_t L3X = 512; // 512
	static constexpr size_t L3Z = 768; // 768
	static constexpr size_t LEFTSIZEL3 = L3Y * L3X;
	static constexpr size_t RIGHTSIZEL3 = L3X * L3Z;

	static constexpr size_t L2Y = 768; // 768
	static constexpr size_t L2X = 256; // 256
	static constexpr size_t L2Z = 768; // 768
	static constexpr size_t LEFTSIZEL2 = L2Y * L2X;
	static constexpr size_t RIGHTSIZEL2 = L2X * L2Z;

	static constexpr size_t L1Y = 8; // 8
	static constexpr size_t L1X = 256; // 256
	static constexpr size_t L1Z = 8; // 8

#else

	static constexpr size_t L3Y = 5000;
	static constexpr size_t L3X = 5000;
	static constexpr size_t L3Z = 5000;
	static constexpr size_t LEFTSIZEL3 = L3Y * L3X;
	static constexpr size_t RIGHTSIZEL3 = L3X * L3Z;

	static constexpr size_t L2Y = 64; // 64
	static constexpr size_t L2X = 512; // 512
	static constexpr size_t L2Z = 64; // 64
	static constexpr size_t LEFTSIZEL2 = L2Y * L2X;
	static constexpr size_t RIGHTSIZEL2 = L2X * L2Z;

	static constexpr size_t L1Y = 8; // 8
	static constexpr size_t L1X = 512; // 512
	static constexpr size_t L1Z = 8; // 8

#endif

};

template<>
struct NumTypeSize<double> {

#if MAT_THREAD_NUMS == 1
	static constexpr size_t L3Y = 768; // 768
	static constexpr size_t L3X = 512; // 512
	static constexpr size_t L3Z = 768; // 768
	static constexpr size_t LEFTSIZEL3 = L3Y * L3X;
	static constexpr size_t RIGHTSIZEL3 = L3X * L3Z;

	static constexpr size_t L2Y = 768; // 768
	static constexpr size_t L2X = 256; // 256
	static constexpr size_t L2Z = 768; // 768
	static constexpr size_t LEFTSIZEL2 = L2Y * L2X;
	static constexpr size_t RIGHTSIZEL2 = L2X * L2Z;

	static constexpr size_t L1Y = 8; // 8
	static constexpr size_t L1X = 256; // 256
	static constexpr size_t L1Z = 4; // 4

#else

	static constexpr size_t L3Y = 5000;
	static constexpr size_t L3X = 5000;
	static constexpr size_t L3Z = 5000;
	static constexpr size_t LEFTSIZEL3 = L3Y * L3X;
	static constexpr size_t RIGHTSIZEL3 = L3X * L3Z;

	static constexpr size_t L2Y = 64; // 64
	static constexpr size_t L2X = 256; // 256
	static constexpr size_t L2Z = 64; // 64
	static constexpr size_t LEFTSIZEL2 = L2Y * L2X;
	static constexpr size_t RIGHTSIZEL2 = L2X * L2Z;

	static constexpr size_t L1Y = 8; // 8
	static constexpr size_t L1X = 128; // 128
	static constexpr size_t L1Z = 4; // 4

#endif

};

template <typename U>
struct MatNumType {
	using type = U;
};

template <typename T>
struct MatNumType<mat<T>> {
	using type = T;
};

template <typename NumType>
class vec {
	friend class mat<NumType>;
	/* raii */
public:
	alignas(8) size_t len;
	alignas(8) NumType* pData;

	vec() = delete;
	vec(size_t inLen) noexcept :
		len(inLen), pData(static_cast<NumType*>(malloc(sizeof(NumType)* inLen))) {}

	vec(size_t inLen, NumType element) noexcept :
		len(inLen), pData(static_cast<NumType*>(malloc(sizeof(NumType)* inLen))) {
		if constexpr (MAT_THREAD_NUMS == 1) {
			for (int idx = 0; idx < len; ++idx) {
				pData[idx] = element;
			}
		}
		else {
#pragma omp parallel num_threads(MAT_THREAD_NUMS) if (len > 10000)
			{
#pragma omp for
				for (int idx = 0; idx < len; ++idx) {
					pData[idx] = element;
				}
			} // omp
		}
	}

	template <typename dist>
	vec(size_t inLen, dist& dis) noexcept :
		len(inLen), pData(static_cast<NumType*>(malloc(sizeof(NumType)* inLen))) {
		std::random_device device;
		std::mt19937 gen(device());
		for (int idx = 0; idx < len; ++idx) {
			pData[idx] = dis(gen);
		}
	}

	vec(const vec& inVec) noexcept :
		len(inVec.len), pData(static_cast<NumType*>(malloc(sizeof(NumType)* inVec.len))) {
		memcpy(pData, inVec.pData, sizeof(NumType) * len);
	}

	vec(vec&& inVec) noexcept :
		len(inVec.len), pData(inVec.pData) {
		inVec.pData = nullptr;
	}

	~vec() {
		if (pData != nullptr) {
			free(pData);
		}
		pData = nullptr;
	}

	/* 运算符重载 */
public:
	vec& operator=(const vec& inVec) noexcept {
		if (pData != nullptr && pData != inVec.pData) {
			free(pData);
			pData = static_cast<NumType*>(malloc(sizeof(NumType) * inVec.len));
		}
		len = inVec.len;
		memcpy(pData, inVec.pData, sizeof(NumType) * len);
		return *this;
	}

	vec& operator=(vec&& inVec) noexcept {
		if (pData != nullptr && pData != inVec.pData) {
			free(pData);
		}
		pData = inVec.pData;
		len = inVec.len;
		inVec.pData = nullptr;
		return *this;
	}

	operator mat<NumType>() {
		mat<NumType> result(len, 1, pData);
		pData = nullptr;
		len = 0;
		return result;
	}

	/* tools */
public:
	void print() const noexcept {
		if constexpr (std::is_integral_v<NumType> || std::is_same_v<NumType, char>) {
			for (int i = 0; i < len; ++i) {
				std::cout << pData[i] << " ";
			}
		}
		else {
			for (int i = 0; i < len; ++i) {
				printf("%.2f ", pData[i]);
			}
		}
	}
};

template <typename NumType>
class mat {
	friend class vec<NumType>;
public:
	alignas(8) size_t rows;
	alignas(8) size_t cols;
	alignas(8) size_t len;
	alignas(8) NumType* pData;
	/* raii */
private:
	mat(size_t inRows, size_t inCols, NumType* inPData) noexcept :
		rows(inRows), cols(inCols), len(inRows* inCols), pData(inPData) {}
public:
	mat() = delete;
	mat(size_t inRows, size_t inCols) noexcept :
		rows(inRows), cols(inCols), len(inRows* inCols),
		pData(static_cast<NumType*>(malloc(sizeof(NumType)* len))) {}
	mat(size_t inRows, size_t inCols, NumType element) noexcept :
		rows(inRows), cols(inCols), len(inRows* inCols),
		pData(static_cast<NumType*>(malloc(sizeof(NumType)* len))) {
		if constexpr (MAT_THREAD_NUMS == 1) {
			for (int idx = 0; idx < len; ++idx) {
				pData[idx] = element;
			}
		}
		else {
#pragma omp parallel num_threads(MAT_THREAD_NUMS) if (len > 10000)
			{
#pragma omp for
				for (int idx = 0; idx < len; ++idx) {
					pData[idx] = element;
				}
			} // omp
		}
	}
	template<typename dist>
	mat(size_t inRows, size_t inCols, dist& dis) noexcept :
		rows(inRows), cols(inCols), len(inRows* inCols),
		pData(static_cast<NumType*>(malloc(sizeof(NumType)* len))) {
		std::random_device device;
		std::mt19937 gen(device());
		for (int i = 0; i < len; ++i) {
			pData[i] = dis(gen);
		}
	}

	mat(const mat& inMat) noexcept : rows(inMat.rows), cols(inMat.cols),
		len(inMat.len), pData(static_cast<NumType*>(malloc(sizeof(NumType)* len))) {
		memcpy(pData, inMat.pData, sizeof(NumType) * len);
	}
	mat(mat&& inMat) noexcept :
		rows(inMat.rows), cols(inMat.cols), len(inMat.len),
		pData(inMat.pData) {
		inMat.pData = nullptr;
	}
	~mat() {
		if (pData != nullptr) {
			free(pData);
		}
		pData = nullptr;
	}

	/* 运算符重载 */
public:
	mat& operator =(const mat& inMat) noexcept {
		if (pData != nullptr && pData != inMat.pData) {
			free(pData);
			pData = static_cast<NumType*>(malloc(sizeof(NumType) * inMat.len));
		}
		rows = inMat.rows;
		cols = inMat.cols;
		len = inMat.len;
		memcpy(pData, inMat.pData, sizeof(NumType) * len);
		return *this;
	}

	mat& operator =(mat&& inMat) noexcept {
		if (pData != nullptr && pData != inMat.pData) {
			free(pData);
		}
		rows = inMat.rows;
		cols = inMat.cols;
		len = inMat.len;
		pData = inMat.pData;
		inMat.pData = nullptr;
		return *this;
	}

	friend bool operator== (const mat& left, const mat& right) {
		return (left.rows == right.rows && left.cols == right.cols && left.len == right.len);
	}

#define OPERATOROVERRIDE(SIGN) template<typename LeftT, typename RightT>                          \
	friend mat operator SIGN(const LeftT& left, const RightT& right) {                            \
		if constexpr (std::is_arithmetic_v<LeftT> && std::is_same_v<RightT, mat>) {               \
			mat result = right;                                                                   \
			if constexpr (MAT_THREAD_NUMS == 1) {                                                 \
				for (int i = 0; i < right.len; ++i) {											  \
					result.pData[i] SIGN##= left;													  \
				}																				  \
			}																					  \
			else {																				  \
_Pragma("omp parallel num_threads(MAT_THREAD_NUMS) if (right.len > 10000)")                        \
				{																				  \
_Pragma("omp for")																			      \
					for (int i = 0; i < right.len; ++i) {										  \
						result.pData[i] SIGN##= left;												  \
					}																			  \
				}																				  \
																								  \
			}																					  \
			return result;																		  \
		}																						  \
		else if constexpr (std::is_same_v<LeftT, mat> && std::is_arithmetic_v<RightT>) {		  \
			mat result = left;																	  \
			if constexpr (MAT_THREAD_NUMS == 1) {												  \
				for (int i = 0; i < left.len; ++i) {											  \
					result.pData[i] SIGN##= right;												  \
				}																				  \
			}																					  \
			else {																				  \
_Pragma("omp parallel num_threads(MAT_THREAD_NUMS) if (left.len > 10000)")						  \
				{																				  \
_Pragma("omp for")																			      \
					for (int i = 0; i < left.len; ++i) {										  \
						result.pData[i] SIGN##= right;											  \
					}																			  \
				}																				  \
																								  \
			}																					  \
			return result;																		  \
		}																						  \
		else if constexpr (std::is_same_v<LeftT, mat> && std::is_same_v<RightT, mat>) {			  \
			if (!(static_cast<mat>(left) == static_cast<mat>(right))) {							  \
				throw std::runtime_error("");													  \
			}																					  \
			mat result(left.rows, left.cols);													  \
			if constexpr (MAT_THREAD_NUMS == 1) {												  \
				for (int i = 0; i < left.len; ++i) {											  \
					result.pData[i] = left.pData[i] SIGN right.pData[i];						  \
				}																				  \
			}																					  \
			else {																				  \
_Pragma("omp parallel num_threads(MAT_THREAD_NUMS) if (left.len > 10000)")           			  \
				{																				  \
_Pragma("omp for")																			      \
					for (int i = 0; i < left.len; ++i) {										  \
						result.pData[i] = left.pData[i] SIGN right.pData[i];					  \
					}																			  \
				}																				  \
			}																					  \
			return result;																		  \
		}																						  \
		return mat(0, 0);																		  \
	}																							  

	OPERATOROVERRIDE(+);
	OPERATOROVERRIDE(-);
#undef OPERATOROVERRIDE

	template <typename LeftT, typename RightT>
	friend mat operator * (const LeftT& left, const RightT& right) {
		if constexpr (std::is_arithmetic_v<LeftT> && std::is_same_v<RightT, mat>) {
			mat result = right;
			if (MAT_THREAD_NUMS == 1) {
				for (int i = 0; i < right.len; ++i) {
					result.pData[i] *= left;
				}
			}
			else {
#pragma omp parallel num_threads(MAT_THREAD_NUMS) if (right.len > 10000)
				{
#pragma omp for
					for (int i = 0; i < right.len; ++i) {
						result.pData[i] *= left;
					}
				}
			}
			return result;
		}
		else if constexpr (std::is_arithmetic_v<RightT> && std::is_same_v<LeftT, mat>) {
			mat result = left;
			if (MAT_THREAD_NUMS == 1) {
				for (int i = 0; i < left.len; ++i) {
					result.pData[i] *= right;
				}
			}
			else {
#pragma omp parallel num_threads(MAT_THREAD_NUMS) if (left.len > 10000)
				{
#pragma omp for
					for (int i = 0; i < left.len; ++i) {
						result.pData[i] *= right;
					}
				}
			}
			return result;
		}
		else if constexpr (std::is_same_v<LeftT, mat> && std::is_same_v<RightT, mat>) {
			const size_t& p = left.rows;
			const size_t& q = left.cols;
			const size_t& r = right.rows;
			const size_t& s = right.cols;

			if (q != r) {
				throw std::runtime_error("");
			}
			mat result(p, s, static_cast<NumType>(0));

			NumType* LEFTBLOCKL3POOL = static_cast<NumType*>(malloc(sizeof(NumType) * NumTypeSize<NumType>::LEFTSIZEL3));
			mat leftBlockL3(0, 0, LEFTBLOCKL3POOL);
			NumType* RIGHTBLOCKL3POOL = static_cast<NumType*>(malloc(sizeof(NumType) * NumTypeSize<NumType>::RIGHTSIZEL3));
			mat rightBlockL3(0, 0, RIGHTBLOCKL3POOL);
			NumType* LEFTBLOCKL2POOL = static_cast<NumType*> (malloc(MAT_THREAD_NUMS * sizeof(NumType) * NumTypeSize<NumType>::LEFTSIZEL2));
			NumType* RIGHTBLOCKL2POOL = static_cast<NumType*>(malloc(MAT_THREAD_NUMS * sizeof(NumType) * NumTypeSize<NumType>::RIGHTSIZEL2));

			if constexpr (MAT_THREAD_NUMS == 1) {
				for (size_t i = 0; i < p; i += NumTypeSize<NumType>::L3Y) {
					leftBlockL3.rows =
						NumTypeSize<NumType>::L3Y > p - i ? p - i : NumTypeSize<NumType>::L3Y;
					for (size_t k = 0; k < q; k += NumTypeSize<NumType>::L3X) {
						rightBlockL3.rows = leftBlockL3.cols =
							NumTypeSize<NumType>::L3X > q - k ? q - k : NumTypeSize<NumType>::L3X;

						/* pack L3 with L2 memLayout */
						size_t tmpLeftBlockL2Rows = 0;
						size_t tmpLeftBlockL2Cols = 0;
						size_t tmpLeftBlockL2Len = 0;
						for (size_t ii = 0; ii < leftBlockL3.rows; ii += NumTypeSize<NumType>::L2Y) {
							tmpLeftBlockL2Rows =
								NumTypeSize<NumType>::L2Y > leftBlockL3.rows - ii ? leftBlockL3.rows - ii : NumTypeSize<NumType>::L2Y;
							for (size_t kk = 0; kk < leftBlockL3.cols; kk += NumTypeSize<NumType>::L2X) {
								tmpLeftBlockL2Cols =
									NumTypeSize<NumType>::L2X > leftBlockL3.cols - kk ? leftBlockL3.cols - kk : NumTypeSize<NumType>::L2X;

								/* Remap leftBlockL3 with leftBlockL2 memlayout in col major */
								for (size_t iii = 0; iii < tmpLeftBlockL2Rows; ++iii) {
									for (size_t kkk = 0; kkk < tmpLeftBlockL2Cols; ++kkk) {
										leftBlockL3.pData[kkk * tmpLeftBlockL2Rows + iii + tmpLeftBlockL2Len] =
											left.pData[(i + ii + iii) * q + k + kk + kkk];
									}
								}

								tmpLeftBlockL2Len += tmpLeftBlockL2Rows * tmpLeftBlockL2Cols;
							}
						}

						for (size_t j = 0; j < s; j += NumTypeSize<NumType>::L3Z) {
							rightBlockL3.cols =
								NumTypeSize<NumType>::L3Z > s - j ? s - j : NumTypeSize<NumType>::L3Z;

							/* pack L3 with L2 memLayout */
							size_t tmpRightBlockL2Rows = 0;
							size_t tmpRightBlockL2Cols = 0;
							size_t tmpRightBlockL2Len = 0;
							for (size_t kk = 0; kk < rightBlockL3.rows; kk += NumTypeSize<NumType>::L2X) {
								tmpRightBlockL2Rows =
									NumTypeSize<NumType>::L2X > rightBlockL3.rows - kk ? rightBlockL3.rows - kk : NumTypeSize<NumType>::L2X;
								for (size_t jj = 0; jj < rightBlockL3.cols; jj += NumTypeSize<NumType>::L2Z) {
									tmpRightBlockL2Cols =
										NumTypeSize<NumType>::L2Z > rightBlockL3.cols - jj ? rightBlockL3.cols - jj : NumTypeSize<NumType>::L2Z;

									/* Remap rightBlockL3 with rightBlockL2 memlayout in row major */
									for (size_t kkk = 0; kkk < tmpRightBlockL2Rows; ++kkk) {
										/* 99.99999% 的概率 MSVC 这里用了向量化加速 */
										memcpy(&rightBlockL3.pData[kkk * tmpRightBlockL2Cols + tmpRightBlockL2Len],
											&right.pData[(k + kk + kkk) * s + j + jj], sizeof(NumType) * tmpRightBlockL2Cols);
									}

									tmpRightBlockL2Len += tmpRightBlockL2Rows * tmpRightBlockL2Cols;
								}
							}

							mat leftBlockL2(0, 0, LEFTBLOCKL2POOL);
							tmpLeftBlockL2Len = 0;
							mat rightBlockL2(0, 0, RIGHTBLOCKL2POOL);
							tmpRightBlockL2Len = 0;
							for (size_t ii = 0; ii < leftBlockL3.rows; ii += NumTypeSize<NumType>::L2Y) {
								leftBlockL2.rows =
									NumTypeSize<NumType>::L2Y > leftBlockL3.rows - ii ? leftBlockL3.rows - ii : NumTypeSize<NumType>::L2Y;

								tmpRightBlockL2Len = 0;
								for (size_t kk = 0; kk < leftBlockL3.cols; kk += NumTypeSize<NumType>::L2X) {
									rightBlockL2.rows = leftBlockL2.cols =
										NumTypeSize<NumType>::L2X > leftBlockL3.cols - kk ? leftBlockL3.cols - kk : NumTypeSize<NumType>::L2X;
									/* pack L2 with L1 memlayout */
									size_t tmpLeftBlockL1Rows = 0;
									size_t tmpLeftBlockL1Cols = 0;
									size_t tmpLeftBlockL1Len = 0;
									for (size_t iii = 0; iii < leftBlockL2.rows; iii += NumTypeSize<NumType>::L1Y) {
										tmpLeftBlockL1Rows =
											NumTypeSize<NumType>::L1Y > leftBlockL2.rows - iii ? leftBlockL2.rows - iii : NumTypeSize<NumType>::L1Y;
										for (size_t kkk = 0; kkk < leftBlockL2.cols; kkk += NumTypeSize<NumType>::L1X) {
											tmpLeftBlockL1Cols =
												NumTypeSize<NumType>::L1X > leftBlockL2.cols - kkk ? leftBlockL2.cols - kkk : NumTypeSize<NumType>::L1X;

											/* col major L2 to col major L1 */
											for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
												memcpy(&leftBlockL2.pData[kkkk * tmpLeftBlockL1Rows + tmpLeftBlockL1Len],
													&leftBlockL3.pData[(kkk + kkkk) * leftBlockL2.rows + iii + tmpLeftBlockL2Len], sizeof(NumType) * tmpLeftBlockL1Rows);
											}

											tmpLeftBlockL1Len += tmpLeftBlockL1Rows * tmpLeftBlockL1Cols;
										}
									}

									tmpLeftBlockL2Len += leftBlockL2.rows * leftBlockL2.cols;

									for (size_t jj = 0; jj < rightBlockL3.cols; jj += NumTypeSize<NumType>::L2Z) {
										rightBlockL2.cols =
											NumTypeSize<NumType>::L2Z > rightBlockL3.cols - jj ? rightBlockL3.cols - jj : NumTypeSize<NumType>::L2Z;
										/* pack L2 with L1 memlayout */
										size_t tmpRightBlockL1Rows = 0;
										size_t tmpRightBlockL1Cols = 0;
										size_t tmpRightBlockL1Len = 0;
										for (size_t kkk = 0; kkk < rightBlockL2.rows; kkk += NumTypeSize<NumType>::L1X) {
											tmpRightBlockL1Rows =
												NumTypeSize<NumType>::L1X > rightBlockL2.rows - kkk ? rightBlockL2.rows - kkk : NumTypeSize<NumType>::L1X;

											for (size_t jjj = 0; jjj < rightBlockL2.cols; jjj += NumTypeSize<NumType>::L1Z) {
												tmpRightBlockL1Cols =
													NumTypeSize<NumType>::L1Z > rightBlockL2.cols - jjj ? rightBlockL2.cols - jjj : NumTypeSize<NumType>::L1Z;

												/* row major L2 to row major L1 */
												for (size_t kkkk = 0; kkkk < tmpRightBlockL1Rows; ++kkkk) {
													memcpy(&rightBlockL2.pData[kkkk * tmpRightBlockL1Cols + tmpRightBlockL1Len],
														&rightBlockL3.pData[(kkkk + kkk) * rightBlockL2.cols + jjj + tmpRightBlockL2Len], sizeof(NumType) * tmpRightBlockL1Cols);
												}

												tmpRightBlockL1Len += tmpRightBlockL1Rows * tmpRightBlockL1Cols;
											}
										}
										tmpRightBlockL2Len += rightBlockL2.rows * rightBlockL2.cols;

										tmpLeftBlockL1Rows = 0;
										tmpLeftBlockL1Cols = 0;
										tmpLeftBlockL1Len = 0;
										for (size_t iii = 0; iii < leftBlockL2.rows; iii += NumTypeSize<NumType>::L1Y) {
											tmpLeftBlockL1Rows =
												NumTypeSize<NumType>::L1Y > leftBlockL2.rows - iii ? leftBlockL2.rows - iii : NumTypeSize<NumType>::L1Y;
											tmpRightBlockL1Rows = 0;
											tmpRightBlockL1Cols = 0;
											tmpRightBlockL1Len = 0;
											for (size_t kkk = 0; kkk < leftBlockL2.cols; kkk += NumTypeSize<NumType>::L1X) {
												tmpRightBlockL1Rows = tmpLeftBlockL1Cols =
													NumTypeSize<NumType>::L1X > leftBlockL2.cols - kkk ? leftBlockL2.cols - kkk : NumTypeSize<NumType>::L1X;

												NumType* __restrict leftPtr = leftBlockL2.pData + tmpLeftBlockL1Len;
												for (size_t jjj = 0; jjj < rightBlockL2.cols; jjj += NumTypeSize<NumType>::L1Z) {
													tmpRightBlockL1Cols =
														NumTypeSize<NumType>::L1Z > rightBlockL2.cols - jjj ? rightBlockL2.cols - jjj : NumTypeSize<NumType>::L1Z;
													NumType* __restrict rightPtr = rightBlockL2.pData + tmpRightBlockL1Len;
													/* kernel */
													if constexpr (std::is_same_v<NumType, float>) {
														if (tmpLeftBlockL1Rows == NumTypeSize<NumType>::L1Y &&
															tmpRightBlockL1Cols == NumTypeSize<NumType>::L1Z) {
															float* __restrict leftBasePtr = leftPtr;
															float* __restrict rightBasePtr = rightPtr;

															size_t resultBasePtrIndex = (i + ii + iii) * s + j + jj + jjj;
															float* resultBasePtr = &result.pData[resultBasePtrIndex];

															__m256 c0 = _mm256_loadu_ps(resultBasePtr);
															__m256 c1 = _mm256_loadu_ps(resultBasePtr + s);
															__m256 c2 = _mm256_loadu_ps(resultBasePtr + 2 * s);
															__m256 c3 = _mm256_loadu_ps(resultBasePtr + 3 * s);
															__m256 c4 = _mm256_loadu_ps(resultBasePtr + 4 * s);
															__m256 c5 = _mm256_loadu_ps(resultBasePtr + 5 * s);
															__m256 c6 = _mm256_loadu_ps(resultBasePtr + 6 * s);
															__m256 c7 = _mm256_loadu_ps(resultBasePtr + 7 * s);

															size_t kkkk = 0;
															for (; kkkk + 3 < tmpLeftBlockL1Cols; kkkk += 4) {
																__m256 b0 = _mm256_loadu_ps(rightBasePtr);
																__m256 b1 = _mm256_loadu_ps(rightBasePtr + NumTypeSize<NumType>::L1Z);
																__m256 b2 = _mm256_loadu_ps(rightBasePtr + 2 * NumTypeSize<NumType>::L1Z);
																__m256 b3 = _mm256_loadu_ps(rightBasePtr + 3 * NumTypeSize<NumType>::L1Z);

																__m256 a0 = _mm256_broadcast_ss(leftBasePtr);
																__m256 a1 = _mm256_broadcast_ss(leftBasePtr + 1);
																__m256 a2 = _mm256_broadcast_ss(leftBasePtr + 2);
																__m256 a3 = _mm256_broadcast_ss(leftBasePtr + 3);
																__m256 a4 = _mm256_broadcast_ss(leftBasePtr + 4);
																__m256 a5 = _mm256_broadcast_ss(leftBasePtr + 5);
																__m256 a6 = _mm256_broadcast_ss(leftBasePtr + 6);
																__m256 a7 = _mm256_broadcast_ss(leftBasePtr + 7);

																c0 = _mm256_fmadd_ps(a0, b0, c0);
																c1 = _mm256_fmadd_ps(a1, b0, c1);
																c2 = _mm256_fmadd_ps(a2, b0, c2);
																c3 = _mm256_fmadd_ps(a3, b0, c3);
																c4 = _mm256_fmadd_ps(a4, b0, c4);
																c5 = _mm256_fmadd_ps(a5, b0, c5);
																c6 = _mm256_fmadd_ps(a6, b0, c6);
																c7 = _mm256_fmadd_ps(a7, b0, c7);

																leftBasePtr += NumTypeSize<NumType>::L1Y;
																BROADCASTANDFMADDSF(b1);
																BROADCASTANDFMADDSF(b2);
																BROADCASTANDFMADDSF(b3);
																rightBasePtr += 4 * NumTypeSize<NumType>::L1Z;

															}
															for (; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																__m256 b0 = _mm256_loadu_ps(rightBasePtr);

																__m256 a0 = _mm256_broadcast_ss(leftBasePtr);
																__m256 a1 = _mm256_broadcast_ss(leftBasePtr + 1);
																__m256 a2 = _mm256_broadcast_ss(leftBasePtr + 2);
																__m256 a3 = _mm256_broadcast_ss(leftBasePtr + 3);
																__m256 a4 = _mm256_broadcast_ss(leftBasePtr + 4);
																__m256 a5 = _mm256_broadcast_ss(leftBasePtr + 5);
																__m256 a6 = _mm256_broadcast_ss(leftBasePtr + 6);
																__m256 a7 = _mm256_broadcast_ss(leftBasePtr + 7);

																c0 = _mm256_fmadd_ps(a0, b0, c0);
																c1 = _mm256_fmadd_ps(a1, b0, c1);
																c2 = _mm256_fmadd_ps(a2, b0, c2);
																c3 = _mm256_fmadd_ps(a3, b0, c3);
																c4 = _mm256_fmadd_ps(a4, b0, c4);
																c5 = _mm256_fmadd_ps(a5, b0, c5);
																c6 = _mm256_fmadd_ps(a6, b0, c6);
																c7 = _mm256_fmadd_ps(a7, b0, c7);

																leftBasePtr += NumTypeSize<NumType>::L1Y;
																rightBasePtr += NumTypeSize<NumType>::L1Z;
															}

															_mm256_storeu_ps(resultBasePtr, c0);
															_mm256_storeu_ps(resultBasePtr + s, c1);
															_mm256_storeu_ps(resultBasePtr + 2 * s, c2);
															_mm256_storeu_ps(resultBasePtr + 3 * s, c3);
															_mm256_storeu_ps(resultBasePtr + 4 * s, c4);
															_mm256_storeu_ps(resultBasePtr + 5 * s, c5);
															_mm256_storeu_ps(resultBasePtr + 6 * s, c6);
															_mm256_storeu_ps(resultBasePtr + 7 * s, c7);
														}
														else {
															size_t resultIOffset = (i + ii + iii) * s;
															size_t resultJOffset = j + jj + jjj;
															for (size_t iiii = 0; iiii < tmpLeftBlockL1Rows; ++iiii) {
																size_t resultIOffsetLoop = iiii * s;
																size_t resultTotalOffset = resultIOffset + resultIOffsetLoop + resultJOffset;
																for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																	size_t leftStrideOffset = kkkk * tmpLeftBlockL1Rows + iiii;
																	size_t rightStrideOffset = kkkk * tmpRightBlockL1Cols;
																	for (size_t jjjj = 0; jjjj < tmpRightBlockL1Cols; ++jjjj) {
																		result.pData[resultTotalOffset + jjjj] +=
																			leftPtr[leftStrideOffset] *
																			rightPtr[rightStrideOffset + jjjj];
																	}
																}
															}
														}
													}
													else if constexpr (std::is_same_v<NumType, double>) {
														if (tmpLeftBlockL1Rows == NumTypeSize<NumType>::L1Y &&
															tmpRightBlockL1Cols == NumTypeSize<NumType>::L1Z) {
															double* __restrict leftBasePtr = leftPtr;
															double* __restrict rightBasePtr = rightPtr;

															size_t resultBasePtrIndex = (i + ii + iii) * s + j + jj + jjj;
															double* resultBasePtr = &result.pData[resultBasePtrIndex];

															__m256d c0 = _mm256_loadu_pd(resultBasePtr);
															__m256d c1 = _mm256_loadu_pd(resultBasePtr + s);
															__m256d c2 = _mm256_loadu_pd(resultBasePtr + 2 * s);
															__m256d c3 = _mm256_loadu_pd(resultBasePtr + 3 * s);
															__m256d c4 = _mm256_loadu_pd(resultBasePtr + 4 * s);
															__m256d c5 = _mm256_loadu_pd(resultBasePtr + 5 * s);
															__m256d c6 = _mm256_loadu_pd(resultBasePtr + 6 * s);
															__m256d c7 = _mm256_loadu_pd(resultBasePtr + 7 * s);

															size_t kkkk = 0;
															for (; kkkk + 3 < tmpLeftBlockL1Cols; kkkk += 4) {

																__m256d b0 = _mm256_loadu_pd(rightBasePtr);
																__m256d b1 = _mm256_loadu_pd(rightBasePtr + NumTypeSize<NumType>::L1Z);
																__m256d b2 = _mm256_loadu_pd(rightBasePtr + 2 * NumTypeSize<NumType>::L1Z);
																__m256d b3 = _mm256_loadu_pd(rightBasePtr + 3 * NumTypeSize<NumType>::L1Z);

																__m256d a0 = _mm256_broadcast_sd(leftBasePtr);
																__m256d a1 = _mm256_broadcast_sd(leftBasePtr + 1);
																__m256d a2 = _mm256_broadcast_sd(leftBasePtr + 2);
																__m256d a3 = _mm256_broadcast_sd(leftBasePtr + 3);
																__m256d a4 = _mm256_broadcast_sd(leftBasePtr + 4);
																__m256d a5 = _mm256_broadcast_sd(leftBasePtr + 5);
																__m256d a6 = _mm256_broadcast_sd(leftBasePtr + 6);
																__m256d a7 = _mm256_broadcast_sd(leftBasePtr + 7);

																c0 = _mm256_fmadd_pd(a0, b0, c0);
																c1 = _mm256_fmadd_pd(a1, b0, c1);
																c2 = _mm256_fmadd_pd(a2, b0, c2);
																c3 = _mm256_fmadd_pd(a3, b0, c3);
																c4 = _mm256_fmadd_pd(a4, b0, c4);
																c5 = _mm256_fmadd_pd(a5, b0, c5);
																c6 = _mm256_fmadd_pd(a6, b0, c6);
																c7 = _mm256_fmadd_pd(a7, b0, c7);

																leftBasePtr += NumTypeSize<NumType>::L1Y;
																BROADCASTANDFMADDDF(b1);
																BROADCASTANDFMADDDF(b2);
																BROADCASTANDFMADDDF(b3);

																rightBasePtr += 4 * NumTypeSize<NumType>::L1Z;
															}
															for (; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																__m256d b0 = _mm256_loadu_pd(rightBasePtr);

																__m256d a0 = _mm256_broadcast_sd(leftBasePtr);
																__m256d a1 = _mm256_broadcast_sd(leftBasePtr + 1);
																__m256d a2 = _mm256_broadcast_sd(leftBasePtr + 2);
																__m256d a3 = _mm256_broadcast_sd(leftBasePtr + 3);
																__m256d a4 = _mm256_broadcast_sd(leftBasePtr + 4);
																__m256d a5 = _mm256_broadcast_sd(leftBasePtr + 5);
																__m256d a6 = _mm256_broadcast_sd(leftBasePtr + 6);
																__m256d a7 = _mm256_broadcast_sd(leftBasePtr + 7);

																c0 = _mm256_fmadd_pd(a0, b0, c0);
																c1 = _mm256_fmadd_pd(a1, b0, c1);
																c2 = _mm256_fmadd_pd(a2, b0, c2);
																c3 = _mm256_fmadd_pd(a3, b0, c3);
																c4 = _mm256_fmadd_pd(a4, b0, c4);
																c5 = _mm256_fmadd_pd(a5, b0, c5);
																c6 = _mm256_fmadd_pd(a6, b0, c6);
																c7 = _mm256_fmadd_pd(a7, b0, c7);

																leftBasePtr += NumTypeSize<NumType>::L1Y;
																rightBasePtr += NumTypeSize<NumType>::L1Z;
															}

															_mm256_storeu_pd(resultBasePtr, c0);
															_mm256_storeu_pd(resultBasePtr + s, c1);
															_mm256_storeu_pd(resultBasePtr + 2 * s, c2);
															_mm256_storeu_pd(resultBasePtr + 3 * s, c3);
															_mm256_storeu_pd(resultBasePtr + 4 * s, c4);
															_mm256_storeu_pd(resultBasePtr + 5 * s, c5);
															_mm256_storeu_pd(resultBasePtr + 6 * s, c6);
															_mm256_storeu_pd(resultBasePtr + 7 * s, c7);
														}
														else {
															size_t resultIOffset = (i + ii + iii) * s;
															size_t resultJOffset = j + jj + jjj;
															for (size_t iiii = 0; iiii < tmpLeftBlockL1Rows; ++iiii) {
																size_t resultIOffsetLoop = iiii * s;
																size_t resultTotalOffset = resultIOffset + resultIOffsetLoop + resultJOffset;
																for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																	size_t leftStrideOffset = kkkk * tmpLeftBlockL1Rows + iiii;
																	size_t rightStrideOffset = kkkk * tmpRightBlockL1Cols;
																	for (size_t jjjj = 0; jjjj < tmpRightBlockL1Cols; ++jjjj) {
																		result.pData[resultTotalOffset + jjjj] +=
																			leftPtr[leftStrideOffset] *
																			rightPtr[rightStrideOffset + jjjj];
																	}
																}
															}
														}
													}

													tmpRightBlockL1Len += tmpRightBlockL1Rows * tmpRightBlockL1Cols;
												}

												tmpLeftBlockL1Len += tmpLeftBlockL1Rows * tmpLeftBlockL1Cols;
											}
										}
									}
								}
							}

							leftBlockL2.pData = nullptr;
							rightBlockL2.pData = nullptr;
						}
					}
				}
			}
			else {

				/* 多线程 */
				std::unordered_map<size_t, size_t> iiMapstoRow;

				for (size_t i = 0; i < p; i += NumTypeSize<NumType>::L3Y) {
					leftBlockL3.rows =
						NumTypeSize<NumType>::L3Y > p - i ? p - i : NumTypeSize<NumType>::L3Y;
					for (size_t k = 0; k < q; k += NumTypeSize<NumType>::L3X) {
						rightBlockL3.rows = leftBlockL3.cols =
							NumTypeSize<NumType>::L3X > q - k ? q - k : NumTypeSize<NumType>::L3X;

						iiMapstoRow.clear();
						/* pack L3 with L2 memLayout */
						size_t tmpLeftBlockL2Rows = 0;
						size_t tmpLeftBlockL2Cols = 0;
						size_t tmpLeftBlockL2Len = 0;
						for (size_t ii = 0; ii < leftBlockL3.rows; ii += NumTypeSize<NumType>::L2Y) {
							iiMapstoRow[ii] = tmpLeftBlockL2Len;
							tmpLeftBlockL2Rows =
								NumTypeSize<NumType>::L2Y > leftBlockL3.rows - ii ? leftBlockL3.rows - ii : NumTypeSize<NumType>::L2Y;
							for (size_t kk = 0; kk < leftBlockL3.cols; kk += NumTypeSize<NumType>::L2X) {
								tmpLeftBlockL2Cols =
									NumTypeSize<NumType>::L2X > leftBlockL3.cols - kk ? leftBlockL3.cols - kk : NumTypeSize<NumType>::L2X;

								/* Remap leftBlockL3 with leftBlockL2 memlayout in col major */
								for (size_t iii = 0; iii < tmpLeftBlockL2Rows; ++iii) {
									for (size_t kkk = 0; kkk < tmpLeftBlockL2Cols; ++kkk) {
										leftBlockL3.pData[kkk * tmpLeftBlockL2Rows + iii + tmpLeftBlockL2Len] =
											left.pData[(i + ii + iii) * q + k + kk + kkk];
									}
								}

								tmpLeftBlockL2Len += tmpLeftBlockL2Rows * tmpLeftBlockL2Cols;
							}
						}

						for (size_t j = 0; j < s; j += NumTypeSize<NumType>::L3Z) {
							rightBlockL3.cols =
								NumTypeSize<NumType>::L3Z > s - j ? s - j : NumTypeSize<NumType>::L3Z;

							/* pack L3 with L2 memLayout */
							size_t tmpRightBlockL2Rows = 0;
							size_t tmpRightBlockL2Cols = 0;
							size_t tmpRightBlockL2Len = 0;
							for (size_t kk = 0; kk < rightBlockL3.rows; kk += NumTypeSize<NumType>::L2X) {
								tmpRightBlockL2Rows =
									NumTypeSize<NumType>::L2X > rightBlockL3.rows - kk ? rightBlockL3.rows - kk : NumTypeSize<NumType>::L2X;
								for (size_t jj = 0; jj < rightBlockL3.cols; jj += NumTypeSize<NumType>::L2Z) {
									tmpRightBlockL2Cols =
										NumTypeSize<NumType>::L2Z > rightBlockL3.cols - jj ? rightBlockL3.cols - jj : NumTypeSize<NumType>::L2Z;

									/* Remap rightBlockL3 with rightBlockL2 memlayout in row major */
									for (size_t kkk = 0; kkk < tmpRightBlockL2Rows; ++kkk) {
										/* 99.99999% 的概率 MSVC 这里用了向量化加速 */
										memcpy(&rightBlockL3.pData[kkk * tmpRightBlockL2Cols + tmpRightBlockL2Len],
											&right.pData[(k + kk + kkk) * s + j + jj], sizeof(NumType) * tmpRightBlockL2Cols);
									}

									tmpRightBlockL2Len += tmpRightBlockL2Rows * tmpRightBlockL2Cols;
								}
							}
#pragma omp parallel num_threads(MAT_THREAD_NUMS) /* 这里不像其他函数有 runtime if 判断, 以此避免性能损失 */
							{
								int threadID = omp_get_thread_num();
								mat leftBlockL2(0, 0, LEFTBLOCKL2POOL + threadID * NumTypeSize<NumType>::LEFTSIZEL2);
								size_t tmpLeftBlockL2Len = 0;  /* 覆盖了原来的 tmpLeftBlockL2Len */
								mat rightBlockL2(0, 0, RIGHTBLOCKL2POOL + threadID * NumTypeSize<NumType>::RIGHTSIZEL2);
								size_t tmpRightBlockL2Len = 0; /* 覆盖了原来的 tmpRightBlockL2Len */
#pragma omp for
								for (int64_t ii = 0; ii < (int64_t)leftBlockL3.rows; ii += NumTypeSize<NumType>::L2Y) {
									leftBlockL2.rows =
										NumTypeSize<NumType>::L2Y > leftBlockL3.rows - ii ? leftBlockL3.rows - ii : NumTypeSize<NumType>::L2Y;
									tmpLeftBlockL2Len = iiMapstoRow[ii];

									tmpRightBlockL2Len = 0;
									for (size_t kk = 0; kk < leftBlockL3.cols; kk += NumTypeSize<NumType>::L2X) {
										rightBlockL2.rows = leftBlockL2.cols =
											NumTypeSize<NumType>::L2X > leftBlockL3.cols - kk ? leftBlockL3.cols - kk : NumTypeSize<NumType>::L2X;
										/* pack L2 with L1 memlayout */
										size_t tmpLeftBlockL1Rows = 0;
										size_t tmpLeftBlockL1Cols = 0;
										size_t tmpLeftBlockL1Len = 0;
										for (size_t iii = 0; iii < leftBlockL2.rows; iii += NumTypeSize<NumType>::L1Y) {
											tmpLeftBlockL1Rows =
												NumTypeSize<NumType>::L1Y > leftBlockL2.rows - iii ? leftBlockL2.rows - iii : NumTypeSize<NumType>::L1Y;
											for (size_t kkk = 0; kkk < leftBlockL2.cols; kkk += NumTypeSize<NumType>::L1X) {
												tmpLeftBlockL1Cols =
													NumTypeSize<NumType>::L1X > leftBlockL2.cols - kkk ? leftBlockL2.cols - kkk : NumTypeSize<NumType>::L1X;

												/* col major L2 to col major L1 */
												for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
													memcpy(&leftBlockL2.pData[kkkk * tmpLeftBlockL1Rows + tmpLeftBlockL1Len],
														&leftBlockL3.pData[(kkk + kkkk) * leftBlockL2.rows + iii + tmpLeftBlockL2Len], sizeof(NumType) * tmpLeftBlockL1Rows);
												}

												tmpLeftBlockL1Len += tmpLeftBlockL1Rows * tmpLeftBlockL1Cols;
											}
										}

										tmpLeftBlockL2Len += leftBlockL2.rows * leftBlockL2.cols;

										for (size_t jj = 0; jj < rightBlockL3.cols; jj += NumTypeSize<NumType>::L2Z) {
											rightBlockL2.cols =
												NumTypeSize<NumType>::L2Z > rightBlockL3.cols - jj ? rightBlockL3.cols - jj : NumTypeSize<NumType>::L2Z;
											/* pack L2 with L1 memlayout */
											size_t tmpRightBlockL1Rows = 0;
											size_t tmpRightBlockL1Cols = 0;
											size_t tmpRightBlockL1Len = 0;
											for (size_t kkk = 0; kkk < rightBlockL2.rows; kkk += NumTypeSize<NumType>::L1X) {
												tmpRightBlockL1Rows =
													NumTypeSize<NumType>::L1X > rightBlockL2.rows - kkk ? rightBlockL2.rows - kkk : NumTypeSize<NumType>::L1X;

												for (size_t jjj = 0; jjj < rightBlockL2.cols; jjj += NumTypeSize<NumType>::L1Z) {
													tmpRightBlockL1Cols =
														NumTypeSize<NumType>::L1Z > rightBlockL2.cols - jjj ? rightBlockL2.cols - jjj : NumTypeSize<NumType>::L1Z;

													/* row major L2 to row major L1 */
													for (size_t kkkk = 0; kkkk < tmpRightBlockL1Rows; ++kkkk) {
														memcpy(&rightBlockL2.pData[kkkk * tmpRightBlockL1Cols + tmpRightBlockL1Len],
															&rightBlockL3.pData[(kkkk + kkk) * rightBlockL2.cols + jjj + tmpRightBlockL2Len], sizeof(NumType) * tmpRightBlockL1Cols);
													}

													tmpRightBlockL1Len += tmpRightBlockL1Rows * tmpRightBlockL1Cols;
												}
											}
											tmpRightBlockL2Len += rightBlockL2.rows * rightBlockL2.cols;

											tmpLeftBlockL1Rows = 0;
											tmpLeftBlockL1Cols = 0;
											tmpLeftBlockL1Len = 0;
											for (size_t iii = 0; iii < leftBlockL2.rows; iii += NumTypeSize<NumType>::L1Y) {
												tmpLeftBlockL1Rows =
													NumTypeSize<NumType>::L1Y > leftBlockL2.rows - iii ? leftBlockL2.rows - iii : NumTypeSize<NumType>::L1Y;
												tmpRightBlockL1Rows = 0;
												tmpRightBlockL1Cols = 0;
												tmpRightBlockL1Len = 0;
												for (size_t kkk = 0; kkk < leftBlockL2.cols; kkk += NumTypeSize<NumType>::L1X) {
													tmpRightBlockL1Rows = tmpLeftBlockL1Cols =
														NumTypeSize<NumType>::L1X > leftBlockL2.cols - kkk ? leftBlockL2.cols - kkk : NumTypeSize<NumType>::L1X;

													NumType* __restrict leftPtr = leftBlockL2.pData + tmpLeftBlockL1Len;
													for (size_t jjj = 0; jjj < rightBlockL2.cols; jjj += NumTypeSize<NumType>::L1Z) {
														tmpRightBlockL1Cols =
															NumTypeSize<NumType>::L1Z > rightBlockL2.cols - jjj ? rightBlockL2.cols - jjj : NumTypeSize<NumType>::L1Z;
														NumType* __restrict rightPtr = rightBlockL2.pData + tmpRightBlockL1Len;
														/* kernel */
														if constexpr (std::is_same_v<NumType, float>) {
															if (tmpLeftBlockL1Rows == NumTypeSize<NumType>::L1Y &&
																tmpRightBlockL1Cols == NumTypeSize<NumType>::L1Z) {
																float* __restrict leftBasePtr = leftPtr;
																float* __restrict rightBasePtr = rightPtr;

																size_t resultBasePtrIndex = (i + ii + iii) * s + j + jj + jjj;
																float* resultBasePtr = &result.pData[resultBasePtrIndex];

																__m256 c0 = _mm256_loadu_ps(resultBasePtr);
																__m256 c1 = _mm256_loadu_ps(resultBasePtr + s);
																__m256 c2 = _mm256_loadu_ps(resultBasePtr + 2 * s);
																__m256 c3 = _mm256_loadu_ps(resultBasePtr + 3 * s);
																__m256 c4 = _mm256_loadu_ps(resultBasePtr + 4 * s);
																__m256 c5 = _mm256_loadu_ps(resultBasePtr + 5 * s);
																__m256 c6 = _mm256_loadu_ps(resultBasePtr + 6 * s);
																__m256 c7 = _mm256_loadu_ps(resultBasePtr + 7 * s);

																size_t kkkk = 0;
																for (; kkkk + 3 < tmpLeftBlockL1Cols; kkkk += 4) {

																	__m256 b0 = _mm256_loadu_ps(rightBasePtr);
																	__m256 b1 = _mm256_loadu_ps(rightBasePtr + NumTypeSize<NumType>::L1Z);
																	__m256 b2 = _mm256_loadu_ps(rightBasePtr + 2 * NumTypeSize<NumType>::L1Z);
																	__m256 b3 = _mm256_loadu_ps(rightBasePtr + 3 * NumTypeSize<NumType>::L1Z);

																	__m256 a0 = _mm256_broadcast_ss(leftBasePtr);
																	__m256 a1 = _mm256_broadcast_ss(leftBasePtr + 1);
																	__m256 a2 = _mm256_broadcast_ss(leftBasePtr + 2);
																	__m256 a3 = _mm256_broadcast_ss(leftBasePtr + 3);
																	__m256 a4 = _mm256_broadcast_ss(leftBasePtr + 4);
																	__m256 a5 = _mm256_broadcast_ss(leftBasePtr + 5);
																	__m256 a6 = _mm256_broadcast_ss(leftBasePtr + 6);
																	__m256 a7 = _mm256_broadcast_ss(leftBasePtr + 7);

																	c0 = _mm256_fmadd_ps(a0, b0, c0);
																	c1 = _mm256_fmadd_ps(a1, b0, c1);
																	c2 = _mm256_fmadd_ps(a2, b0, c2);
																	c3 = _mm256_fmadd_ps(a3, b0, c3);
																	c4 = _mm256_fmadd_ps(a4, b0, c4);
																	c5 = _mm256_fmadd_ps(a5, b0, c5);
																	c6 = _mm256_fmadd_ps(a6, b0, c6);
																	c7 = _mm256_fmadd_ps(a7, b0, c7);

																	leftBasePtr += NumTypeSize<NumType>::L1Y;
																	BROADCASTANDFMADDSF(b1);
																	BROADCASTANDFMADDSF(b2);
																	BROADCASTANDFMADDSF(b3);

																	rightBasePtr += 4 * NumTypeSize<NumType>::L1Z;
																}
																for (; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																	__m256 b0 = _mm256_loadu_ps(rightBasePtr);

																	__m256 a0 = _mm256_broadcast_ss(leftBasePtr);
																	__m256 a1 = _mm256_broadcast_ss(leftBasePtr + 1);
																	__m256 a2 = _mm256_broadcast_ss(leftBasePtr + 2);
																	__m256 a3 = _mm256_broadcast_ss(leftBasePtr + 3);
																	__m256 a4 = _mm256_broadcast_ss(leftBasePtr + 4);
																	__m256 a5 = _mm256_broadcast_ss(leftBasePtr + 5);
																	__m256 a6 = _mm256_broadcast_ss(leftBasePtr + 6);
																	__m256 a7 = _mm256_broadcast_ss(leftBasePtr + 7);

																	c0 = _mm256_fmadd_ps(a0, b0, c0);
																	c1 = _mm256_fmadd_ps(a1, b0, c1);
																	c2 = _mm256_fmadd_ps(a2, b0, c2);
																	c3 = _mm256_fmadd_ps(a3, b0, c3);
																	c4 = _mm256_fmadd_ps(a4, b0, c4);
																	c5 = _mm256_fmadd_ps(a5, b0, c5);
																	c6 = _mm256_fmadd_ps(a6, b0, c6);
																	c7 = _mm256_fmadd_ps(a7, b0, c7);

																	leftBasePtr += NumTypeSize<NumType>::L1Y;
																	rightBasePtr += NumTypeSize<NumType>::L1Z;
																}

																_mm256_storeu_ps(resultBasePtr, c0);
																_mm256_storeu_ps(resultBasePtr + s, c1);
																_mm256_storeu_ps(resultBasePtr + 2 * s, c2);
																_mm256_storeu_ps(resultBasePtr + 3 * s, c3);
																_mm256_storeu_ps(resultBasePtr + 4 * s, c4);
																_mm256_storeu_ps(resultBasePtr + 5 * s, c5);
																_mm256_storeu_ps(resultBasePtr + 6 * s, c6);
																_mm256_storeu_ps(resultBasePtr + 7 * s, c7);
															}
															else {
																size_t resultIOffset = (i + ii + iii) * s;
																size_t resultJOffset = j + jj + jjj;
																for (size_t iiii = 0; iiii < tmpLeftBlockL1Rows; ++iiii) {
																	size_t resultIOffsetLoop = iiii * s;
																	size_t resultTotalOffset = resultIOffset + resultIOffsetLoop + resultJOffset;
																	for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																		size_t leftStrideOffset = kkkk * tmpLeftBlockL1Rows + iiii;
																		size_t rightStrideOffset = kkkk * tmpRightBlockL1Cols;
																		for (size_t jjjj = 0; jjjj < tmpRightBlockL1Cols; ++jjjj) {
																			result.pData[resultTotalOffset + jjjj] +=
																				leftPtr[leftStrideOffset] *
																				rightPtr[rightStrideOffset + jjjj];
																		}
																	}
																}
															}
														}
														else if constexpr (std::is_same_v<NumType, double>) {
															if (tmpLeftBlockL1Rows == NumTypeSize<NumType>::L1Y &&
																tmpRightBlockL1Cols == NumTypeSize<NumType>::L1Z) {
																double* __restrict leftBasePtr = leftPtr;
																double* __restrict rightBasePtr = rightPtr;

																size_t resultBasePtrIndex = (i + ii + iii) * s + j + jj + jjj;
																double* resultBasePtr = &result.pData[resultBasePtrIndex];

																__m256d c0 = _mm256_loadu_pd(resultBasePtr);
																__m256d c1 = _mm256_loadu_pd(resultBasePtr + s);
																__m256d c2 = _mm256_loadu_pd(resultBasePtr + 2 * s);
																__m256d c3 = _mm256_loadu_pd(resultBasePtr + 3 * s);
																__m256d c4 = _mm256_loadu_pd(resultBasePtr + 4 * s);
																__m256d c5 = _mm256_loadu_pd(resultBasePtr + 5 * s);
																__m256d c6 = _mm256_loadu_pd(resultBasePtr + 6 * s);
																__m256d c7 = _mm256_loadu_pd(resultBasePtr + 7 * s);

																size_t kkkk = 0;
																for (; kkkk + 3 < tmpLeftBlockL1Cols; kkkk += 4) {

																	__m256d b0 = _mm256_loadu_pd(rightBasePtr);
																	__m256d b1 = _mm256_loadu_pd(rightBasePtr + NumTypeSize<NumType>::L1Z);
																	__m256d b2 = _mm256_loadu_pd(rightBasePtr + 2 * NumTypeSize<NumType>::L1Z);
																	__m256d b3 = _mm256_loadu_pd(rightBasePtr + 3 * NumTypeSize<NumType>::L1Z);

																	__m256d a0 = _mm256_broadcast_sd(leftBasePtr);
																	__m256d a1 = _mm256_broadcast_sd(leftBasePtr + 1);
																	__m256d a2 = _mm256_broadcast_sd(leftBasePtr + 2);
																	__m256d a3 = _mm256_broadcast_sd(leftBasePtr + 3);
																	__m256d a4 = _mm256_broadcast_sd(leftBasePtr + 4);
																	__m256d a5 = _mm256_broadcast_sd(leftBasePtr + 5);
																	__m256d a6 = _mm256_broadcast_sd(leftBasePtr + 6);
																	__m256d a7 = _mm256_broadcast_sd(leftBasePtr + 7);

																	c0 = _mm256_fmadd_pd(a0, b0, c0);
																	c1 = _mm256_fmadd_pd(a1, b0, c1);
																	c2 = _mm256_fmadd_pd(a2, b0, c2);
																	c3 = _mm256_fmadd_pd(a3, b0, c3);
																	c4 = _mm256_fmadd_pd(a4, b0, c4);
																	c5 = _mm256_fmadd_pd(a5, b0, c5);
																	c6 = _mm256_fmadd_pd(a6, b0, c6);
																	c7 = _mm256_fmadd_pd(a7, b0, c7);

																	leftBasePtr += NumTypeSize<NumType>::L1Y;
																	BROADCASTANDFMADDDF(b1);
																	BROADCASTANDFMADDDF(b2);
																	BROADCASTANDFMADDDF(b3);

																	rightBasePtr += 4 * NumTypeSize<NumType>::L1Z;
																}
																for (; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																	__m256d b0 = _mm256_loadu_pd(rightBasePtr);

																	__m256d a0 = _mm256_broadcast_sd(leftBasePtr);
																	__m256d a1 = _mm256_broadcast_sd(leftBasePtr + 1);
																	__m256d a2 = _mm256_broadcast_sd(leftBasePtr + 2);
																	__m256d a3 = _mm256_broadcast_sd(leftBasePtr + 3);
																	__m256d a4 = _mm256_broadcast_sd(leftBasePtr + 4);
																	__m256d a5 = _mm256_broadcast_sd(leftBasePtr + 5);
																	__m256d a6 = _mm256_broadcast_sd(leftBasePtr + 6);
																	__m256d a7 = _mm256_broadcast_sd(leftBasePtr + 7);

																	c0 = _mm256_fmadd_pd(a0, b0, c0);
																	c1 = _mm256_fmadd_pd(a1, b0, c1);
																	c2 = _mm256_fmadd_pd(a2, b0, c2);
																	c3 = _mm256_fmadd_pd(a3, b0, c3);
																	c4 = _mm256_fmadd_pd(a4, b0, c4);
																	c5 = _mm256_fmadd_pd(a5, b0, c5);
																	c6 = _mm256_fmadd_pd(a6, b0, c6);
																	c7 = _mm256_fmadd_pd(a7, b0, c7);

																	leftBasePtr += NumTypeSize<NumType>::L1Y;
																	rightBasePtr += NumTypeSize<NumType>::L1Z;
																}

																_mm256_storeu_pd(resultBasePtr, c0);
																_mm256_storeu_pd(resultBasePtr + s, c1);
																_mm256_storeu_pd(resultBasePtr + 2 * s, c2);
																_mm256_storeu_pd(resultBasePtr + 3 * s, c3);
																_mm256_storeu_pd(resultBasePtr + 4 * s, c4);
																_mm256_storeu_pd(resultBasePtr + 5 * s, c5);
																_mm256_storeu_pd(resultBasePtr + 6 * s, c6);
																_mm256_storeu_pd(resultBasePtr + 7 * s, c7);
															}
															else {
																size_t resultIOffset = (i + ii + iii) * s;
																size_t resultJOffset = j + jj + jjj;
																for (size_t iiii = 0; iiii < tmpLeftBlockL1Rows; ++iiii) {
																	size_t resultIOffsetLoop = iiii * s;
																	size_t resultTotalOffset = resultIOffset + resultIOffsetLoop + resultJOffset;
																	for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																		size_t leftStrideOffset = kkkk * tmpLeftBlockL1Rows + iiii;
																		size_t rightStrideOffset = kkkk * tmpRightBlockL1Cols;
																		for (size_t jjjj = 0; jjjj < tmpRightBlockL1Cols; ++jjjj) {
																			result.pData[resultTotalOffset + jjjj] +=
																				leftPtr[leftStrideOffset] *
																				rightPtr[rightStrideOffset + jjjj];
																		}
																	}
																}
															}
														}
														else { /* fallback 朴素三循环 */
															size_t resultIOffset = (i + ii + iii) * s;
															size_t resultJOffset = j + jj + jjj;
															for (size_t iiii = 0; iiii < tmpLeftBlockL1Rows; ++iiii) {
																size_t resultIOffsetLoop = iiii * s;
																size_t resultTotalOffset = resultIOffset + resultIOffsetLoop + resultJOffset;
																for (size_t kkkk = 0; kkkk < tmpLeftBlockL1Cols; ++kkkk) {
																	size_t leftStrideOffset = kkkk * tmpLeftBlockL1Rows + iiii;
																	size_t rightStrideOffset = kkkk * tmpRightBlockL1Cols;
																	for (size_t jjjj = 0; jjjj < tmpRightBlockL1Cols; ++jjjj) {
																		result.pData[resultTotalOffset + jjjj] +=
																			leftPtr[leftStrideOffset] *
																			rightPtr[rightStrideOffset + jjjj];
																	}
																}
															}
														}

														tmpRightBlockL1Len += tmpRightBlockL1Rows * tmpRightBlockL1Cols;
													}

													tmpLeftBlockL1Len += tmpLeftBlockL1Rows * tmpLeftBlockL1Cols;
												}
											}
										}
									}
								}

								leftBlockL2.pData = nullptr;
								rightBlockL2.pData = nullptr;
							} // omp parallel section
						}
					}
				}
			}

			free(LEFTBLOCKL3POOL);
			free(RIGHTBLOCKL3POOL);
			free(LEFTBLOCKL2POOL);
			free(RIGHTBLOCKL2POOL);

			leftBlockL3.pData = nullptr;
			rightBlockL3.pData = nullptr;

			return result;
		} // if mat mat
		return mat(0, 0);
	}

	/* tools */
public:
	void print() const {
		if constexpr (std::is_integral_v<NumType> || std::is_same_v<NumType, char>) {
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					std::cout << pData[i * cols + j] << " ";
				}
				std::cout << '\n';
			}
		}
		else {
			for (int i = 0; i < rows; ++i) {
				for (int j = 0; j < cols; ++j) {
					printf("%.2f ", pData[i * cols + j]);
				}
				std::cout << '\n';
			}
		}
	}
};
