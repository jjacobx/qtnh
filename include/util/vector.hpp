#ifndef __UTIL_VECTOR__
#define __UTIL_VECTOR__

#include <vector>

namespace qtnh {
  namespace utils {
    template<typename T>
    std::vector<T> concat_vecs_two(std::vector<T> vec1, std::vector<T> vec2) {
      std::vector<T> vec = vec1;
      vec.insert(vec.end(), vec2.begin(), vec2.end());
      return vec;
    }

    template<typename T, typename... Ts>
    std::vector<T> concat_vecs(std::vector<T> vec1, std::vector<T> vec2, std::vector<Ts>... vecs) {
      const auto NVECS = sizeof...(vecs);
      
      if constexpr (NVECS > 0) {
        return concat_vecs_two(vec1, concat_vecs(vec2, vecs...));
      } else {
        return concat_vecs_two(vec1, vec2);
      }
    }

    template<typename T, std::size_t N>
    std::vector<T> concat_vecs(std::array<std::vector<T>, N> vecs) {
      std::vector<T> res(0);
      for (auto i = 0UL; i < N; ++i) {
        res = concat_vecs_two(res, vecs.at(i));
      }

      return res;
    }

    template <typename T>
    std::pair<std::vector<T>, std::vector<T>> split_vec(std::vector<T> vec, std::size_t n) {
      return { std::vector<T>(vec.begin(), vec.begin() + n), std::vector<T>(vec.begin() + n, vec.end()) };
    }

    // TODO: Restrict types: each of Ns must be size_t
    template <typename T, typename... Ns>
    auto split_vec_rel(std::vector<T> vec, Ns... ns) {
      const auto NSPLITS = sizeof...(ns);
      std::array<std::size_t, NSPLITS> splits { ns... };
      std::array<std::vector<T>, NSPLITS + 1> vecs;

      auto delta = 0UL;
      for (auto i = 0UL; i < NSPLITS; ++i) {
        vecs.at(i) = std::vector<T>(vec.begin() + delta, vec.begin() + delta + splits.at(i));
        delta += splits.at(i);
      }

      vecs.at(NSPLITS) = std::vector<T>(vec.begin() + delta, vec.end());
      return vecs;
    }

    template <typename T, std::size_t N>
    std::vector<T> arr_to_vec(std::array<T, N> arr) {
      return std::vector<T>(std::begin(arr), std::end(arr));
    }

    template <typename T>
    std::vector<T> permute_vec(std::vector<T> vec, std::vector<std::size_t> ptup) {
      auto vec_perm = vec;
      for (std::size_t i = 0; i < vec.size(); ++i) {
        vec_perm.at(ptup.at(i)) = vec.at(i);
      }

      return vec_perm;
    }

    template <typename T>
    std::vector<T> extract_vec(std::vector<T>& vec, std::size_t i1, std::size_t i2) {
      std::vector<T> ext(vec.begin() + i1, vec.begin() + i2);
      vec.erase(vec.begin() + i1, vec.begin() + i2);

      return ext;
    }

    template <typename T, class BinaryOperation>
    std::vector<T> combine_part(std::vector<T> vec, std::size_t from, std::size_t to, 
                                T init, BinaryOperation op) {
      auto val = std::accumulate(vec.begin() + from, vec.begin() + to + 1, init, op);
      vec.at(from) = val;
      vec.erase(vec.begin() + from + 1, vec.begin() + to + 1);

      return vec;
    }

    template <typename T>
    std::vector<T> vec_incr(T from, T to) {
      std::vector<T> res(to - from + 1);
      std::iota(res.begin(), res.end(), from);
      return res;
    }
  }
}

#endif