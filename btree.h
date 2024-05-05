#ifndef __FC_BTREE_H__
#define __FC_BTREE_H__

#define FC_PREFER_BINARY_SEARCH 0

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>

using namespace std;

namespace frozenca {

using attr_t = int32_t;

template <typename K, typename V> struct BTreePair {
  K first;
  V second;

  BTreePair(K &&k, V &&v) : first(forward<K>(k)), second(forward<V>(v)) {}

  BTreePair() = default;

  BTreePair(K &&k) : first(forward<K>(k)), second() {}

  BTreePair(V &&v) : first(), second(forward<V>(v)) {}

  operator pair<const K &, V &>() noexcept { return {first, second}; }

  friend bool operator==(const BTreePair &lhs, const BTreePair &rhs) noexcept {
    return lhs.first == rhs.first && lhs.second == rhs.second;
  }

  friend bool operator!=(const BTreePair &lhs, const BTreePair &rhs) noexcept {
    return !(lhs == rhs);
  }
};

template <typename T> struct TreePairRef { using type = T &; };

template <typename T, typename U> struct TreePairRef<BTreePair<T, U>> {
  using type = pair<const T &, U &>;
};

template <typename TreePair>
using PairRefType = typename TreePairRef<TreePair>::type;

template <typename T, typename U>
bool operator==(const BTreePair<T, U> &lhs,
                const PairRefType<BTreePair<T, U>> &rhs) noexcept {
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <typename T, typename U>
bool operator!=(const BTreePair<T, U> &lhs,
                const PairRefType<BTreePair<T, U>> &rhs) noexcept {
  return !(lhs == rhs);
}

template <typename T, typename U>
bool operator==(const PairRefType<BTreePair<T, U>> &lhs,
                const BTreePair<T, U> &rhs) noexcept {
  return rhs == lhs;
}

template <typename T, typename U>
bool operator!=(const PairRefType<BTreePair<T, U>> &lhs,
                const BTreePair<T, U> &rhs) noexcept {
  return rhs != lhs;
}

template <typename V> struct Projection {
  const auto &operator()(const V &value) const noexcept { return value.first; }
};

template <typename V> struct ProjectionIter {
  auto &operator()(V &iter_ref) noexcept { return iter_ref.first; }

  const auto &operator()(const V &iter_ref) const noexcept {
    return iter_ref.first;
  }
};

inline static constexpr int32_t Fanout = 64;

template <typename K, typename V, typename Comp, bool AllowDup> class BTreeBase;

template <typename K, typename V, typename Comp, bool AllowDup>
struct join_helper;

template <typename K, typename V, typename Comp, bool AllowDup, typename T>
struct split_helper;

template <typename K, typename V, typename Comp, bool AllowDup>
class BTreeBase {

  struct Node;
  using Alloc = allocator<Node>;

  static_assert((Fanout % (sizeof(K) == 4 ? 8 : 4) == 0) && Fanout <= 128);

  static constexpr bool is_set_ = is_same_v<K, V>;

  static constexpr bool use_linsearch_ =
#if FC_PREFER_BINARY_SEARCH
      is_arithmetic_v<K> && (Fanout <= 32);
#else
      is_arithmetic_v<K> && (Fanout <= 128);
#endif

  static constexpr bool CompIsLess =
      is_same_v<Comp, ranges::less> || is_same_v<Comp, less<K>>;
  static constexpr bool CompIsGreater =
      is_same_v<Comp, ranges::greater> || is_same_v<Comp, greater<K>>;

  struct Node {
    using keys_type = vector<V>;

    keys_type keys_;
    Node *parent_ = nullptr;
    attr_t size_ = 0;
    attr_t index_ = 0;
    attr_t height_ = 0;
    vector<unique_ptr<Node>> children_;

    Node() { keys_.reserve(2 * Fanout); }

    Node(const Node &node) = delete;
    Node &operator=(const Node &node) = delete;
    Node(Node &&node) = delete;
    Node &operator=(Node &&node) = delete;

    [[nodiscard]] bool is_leaf() const noexcept { return children_.empty(); }

    [[nodiscard]] bool is_full() const noexcept {
      return ssize(keys_) == 2 * Fanout - 1;
    }

    [[nodiscard]] bool can_take_key() const noexcept {
      return ssize(keys_) > Fanout - 1;
    }

    [[nodiscard]] bool has_minimal_keys() const noexcept {
      return parent_ && ssize(keys_) == Fanout - 1;
    }

    [[nodiscard]] bool empty() const noexcept { return keys_.empty(); }

    void clear_keys() noexcept { keys_.clear(); }

    [[nodiscard]] attr_t nkeys() const noexcept {
      return static_cast<attr_t>(ssize(keys_));
    }
  };

  struct BTreeNonConstIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = V *;
    using reference = V &;
    using iterator_category = bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(value_type &val) noexcept { return val; }
  };

  struct BTreeConstIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = const V *;
    using reference = const V &;
    using iterator_category = bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(const value_type &val) noexcept { return val; }
  };

  struct BTreeRefIterTraits {
    using difference_type = attr_t;
    using value_type = V;
    using pointer = V *;
    using reference = PairRefType<V>;
    using iterator_category = bidirectional_iterator_tag;
    using iterator_concept = iterator_category;

    static reference make_ref(value_type &val) noexcept {
      return {cref(val.first), ref(val.second)};
    }
  };

  template <typename IterTraits> struct BTreeIterator {
    using difference_type = typename IterTraits::difference_type;
    using value_type = typename IterTraits::value_type;
    using pointer = typename IterTraits::pointer;
    using reference = typename IterTraits::reference;
    using iterator_category = typename IterTraits::iterator_category;
    using iterator_concept = typename IterTraits::iterator_concept;

    Node *node_ = nullptr;
    attr_t index_;

    BTreeIterator() noexcept = default;

    BTreeIterator(Node *node, attr_t i) noexcept : node_{node}, index_{i} {}

    template <typename IterTraitsOther>
    BTreeIterator(const BTreeIterator<IterTraitsOther> &other) noexcept
        : BTreeIterator(other.node_, other.index_) {}

    reference operator*() const noexcept {
      return IterTraits::make_ref(node_->keys_[index_]);
    }

    pointer operator->() const noexcept { return &(node_->keys_[index_]); }

    void climb() noexcept {
      while (node_->parent_ && index_ == node_->nkeys()) {
        index_ = node_->index_;
        node_ = node_->parent_;
      }
    }

    void increment() noexcept {
      if (!node_->is_leaf()) {
        node_ = leftmost_leaf(node_->children_[index_ + 1].get());
        index_ = 0;
      } else {
        ++index_;
        while (node_->parent_ && index_ == node_->nkeys()) {
          index_ = node_->index_;
          node_ = node_->parent_;
        }
      }
    }

    void decrement() noexcept {
      if (!node_->is_leaf()) {
        node_ = rightmost_leaf(node_->children_[index_].get());
        index_ = node_->nkeys() - 1;
      } else if (index_ > 0) {
        --index_;
      } else {
        while (node_->parent_ && node_->index_ == 0) {
          node_ = node_->parent_;
        }
        if (node_->index_ > 0) {
          index_ = node_->index_ - 1;
          node_ = node_->parent_;
        }
      }
    }

    BTreeIterator &operator++() noexcept {
      increment();
      return *this;
    }

    BTreeIterator operator++(int) noexcept {
      BTreeIterator temp = *this;
      increment();
      return temp;
    }

    BTreeIterator &operator--() noexcept {
      decrement();
      return *this;
    }

    BTreeIterator operator--(int) noexcept {
      BTreeIterator temp = *this;
      decrement();
      return temp;
    }

    friend bool operator==(const BTreeIterator &x,
                           const BTreeIterator &y) noexcept {
      return x.node_ == y.node_ && x.index_ == y.index_;
    }

    friend bool operator!=(const BTreeIterator &x,
                           const BTreeIterator &y) noexcept {
      return !(x == y);
    }
  };

public:
  using key_type = K;
  using value_type = V;
  using reference_type = conditional_t<is_set_, const V &, PairRefType<V>>;
  using const_reference_type = const V &;
  using node_type = Node;
  using size_type = size_t;
  using difference_type = attr_t;
  using allocator_type = Alloc;
  using nodeptr_type = unique_ptr<Node>;

  using Proj = conditional_t<is_set_, identity, Projection<const V &>>;
  using ProjIter =
      conditional_t<is_set_, identity, ProjectionIter<PairRefType<V>>>;

  static_assert(indirect_strict_weak_order<
                Comp, projected<ranges::iterator_t<vector<V>>, Proj>>);

private:
  using nonconst_iterator_type = BTreeIterator<BTreeNonConstIterTraits>;

public:
  using iterator_type = BTreeIterator<
      conditional_t<is_set_, BTreeConstIterTraits, BTreeRefIterTraits>>;
  using const_iterator_type = BTreeIterator<BTreeConstIterTraits>;
  using reverse_iterator_type = reverse_iterator<iterator_type>;
  using const_reverse_iterator_type = reverse_iterator<const_iterator_type>;

private:
  nodeptr_type root_;
  const_iterator_type begin_;

protected:
  nodeptr_type make_node() { return make_unique<Node>(); }

public:
  BTreeBase() : root_(make_node()), begin_{root_.get(), 0} {}

  BTreeBase(initializer_list<value_type> init) : BTreeBase() {
    for (auto val : init) {
      insert(move(val));
    }
  }

  BTreeBase(const BTreeBase &other) = delete;
  BTreeBase &operator=(const BTreeBase &other) = delete;
  BTreeBase(BTreeBase &&other) noexcept = default;
  BTreeBase &operator=(BTreeBase &&other) noexcept = default;

  void swap(BTreeBase &other) noexcept {
    swap(root_, other.root_);
    swap(begin_, other.begin_);
  }

  [[nodiscard]] iterator_type begin() noexcept { return begin_; }

  [[nodiscard]] const_iterator_type begin() const noexcept {
    return const_iterator_type(begin_);
  }

  [[nodiscard]] const_iterator_type cbegin() const noexcept {
    return const_iterator_type(begin_);
  }

  [[nodiscard]] iterator_type end() noexcept {
    return iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] const_iterator_type end() const noexcept {
    return const_iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] const_iterator_type cend() const noexcept {
    return const_iterator_type(root_.get(), root_->nkeys());
  }

  [[nodiscard]] reverse_iterator_type rbegin() noexcept {
    return reverse_iterator_type(begin());
  }

  [[nodiscard]] const_reverse_iterator_type rbegin() const noexcept {
    return const_reverse_iterator_type(begin());
  }

  [[nodiscard]] const_reverse_iterator_type crbegin() const noexcept {
    return const_reverse_iterator_type(cbegin());
  }

  [[nodiscard]] reverse_iterator_type rend() noexcept {
    return reverse_iterator_type(end());
  }

  [[nodiscard]] const_reverse_iterator_type rend() const noexcept {
    return const_reverse_iterator_type(end());
  }

  [[nodiscard]] const_reverse_iterator_type crend() const noexcept {
    return const_reverse_iterator_type(cend());
  }

  [[nodiscard]] bool empty() const noexcept { return root_->size_ == 0; }

  [[nodiscard]] size_type size() const noexcept {
    return static_cast<size_type>(root_->size_);
  }

  [[nodiscard]] attr_t height() const noexcept { return root_->height_; }

protected:
  [[nodiscard]] Node *get_root() noexcept { return root_.get(); }

  [[nodiscard]] Node *get_root() const noexcept { return root_.get(); }

public:
  void clear() {
    root_ = make_node();
    begin_ = iterator_type(root_.get(), 0);
  }

protected:
  static Node *rightmost_leaf(Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[ssize(curr->children_) - 1].get();
    }
    return curr;
  }

  static const Node *rightmost_leaf(const Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[ssize(curr->children_) - 1].get();
    }
    return curr;
  }

  static Node *leftmost_leaf(Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[0].get();
    }
    return curr;
  }

  static const Node *leftmost_leaf(const Node *curr) noexcept {
    while (curr && !curr->is_leaf()) {
      curr = curr->children_[0].get();
    }
    return curr;
  }

  void promote_root_if_necessary() {
    if (root_->empty()) {
      root_ = move(root_->children_[0]);
      root_->index_ = 0;
      root_->parent_ = nullptr;
    }
  }

  void set_begin() { begin_ = iterator_type(leftmost_leaf(root_.get()), 0); }

  void left_rotate(Node *node) {
    auto parent = node->parent_;
    auto sibling = parent->children_[node->index_ + 1].get();

    node->keys_.push_back(move(parent->keys_[node->index_]));
    parent->keys_[node->index_] = move(sibling->keys_.front());
    shift_left(sibling->keys_.begin(), sibling->keys_.end(), 1);
    sibling->keys_.pop_back();

    node->size_++;
    sibling->size_--;

    if (!node->is_leaf()) {
      const auto orphan_size = sibling->children_.front()->size_;
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      sibling->children_.front()->parent_ = node;
      sibling->children_.front()->index_ =
          static_cast<attr_t>(ssize(node->children_));
      node->children_.push_back(move(sibling->children_.front()));
      shift_left(sibling->children_.begin(), sibling->children_.end(), 1);
      sibling->children_.pop_back();
      for (auto &&child : sibling->children_) {
        child->index_--;
      }
    }
  }

  void left_rotate_n(Node *node, attr_t n) {
    if (n == 1) {
      left_rotate(node);
      return;
    }

    auto parent = node->parent_;
    auto sibling = parent->children_[node->index_ + 1].get();

    node->keys_.push_back(move(parent->keys_[node->index_]));
    ranges::move(sibling->keys_ | views::take(n - 1),
                 back_inserter(node->keys_));
    parent->keys_[node->index_] = move(sibling->keys_[n - 1]);
    shift_left(sibling->keys_.begin(), sibling->keys_.end(), n);
    sibling->keys_.resize(sibling->nkeys() - n);

    node->size_ += n;
    sibling->size_ -= n;

    if (!node->is_leaf()) {
      attr_t orphan_size = 0;
      attr_t immigrant_index = static_cast<attr_t>(ssize(node->children_));
      for (auto &&immigrant : sibling->children_ | views::take(n)) {
        immigrant->parent_ = node;
        immigrant->index_ = immigrant_index++;
        orphan_size += immigrant->size_;
      }
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      ranges::move(sibling->children_ | views::take(n),
                   back_inserter(node->children_));
      shift_left(sibling->children_.begin(), sibling->children_.end(), n);
      for (attr_t idx = 0; idx < n; ++idx) {
        sibling->children_.pop_back();
      }
      attr_t sibling_index = 0;
      for (auto &&child : sibling->children_) {
        child->index_ = sibling_index++;
      }
    }
  }

  void right_rotate(Node *node) {
    auto parent = node->parent_;
    auto sibling = parent->children_[node->index_ - 1].get();

    node->keys_.insert(node->keys_.begin(),
                       move(parent->keys_[node->index_ - 1]));
    parent->keys_[node->index_ - 1] = move(sibling->keys_.back());
    sibling->keys_.pop_back();

    node->size_++;
    sibling->size_--;

    if (!node->is_leaf()) {
      const auto orphan_size = sibling->children_.back()->size_;
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      sibling->children_.back()->parent_ = node;
      sibling->children_.back()->index_ = 0;

      node->children_.insert(node->children_.begin(),
                             move(sibling->children_.back()));
      sibling->children_.pop_back();
      for (auto &&child : node->children_ | views::drop(1)) {
        child->index_++;
      }
    }
  }

  void right_rotate_n(Node *node, attr_t n) {
    if (n == 1) {
      right_rotate(node);
      return;
    }

    auto parent = node->parent_;
    auto sibling = parent->children_[node->index_ - 1].get();

    ranges::move(sibling->keys_ | views::drop(sibling->nkeys() - n) |
                     views::take(n - 1),
                 back_inserter(node->keys_));
    node->keys_.push_back(move(parent->keys_[node->index_ - 1]));
    parent->keys_[node->index_ - 1] = move(sibling->keys_.back());
    ranges::rotate(node->keys_ | views::reverse, node->keys_.rbegin() + n);
    sibling->keys_.resize(sibling->nkeys() - n);

    node->size_ += n;
    sibling->size_ -= n;

    if (!node->is_leaf()) {
      attr_t orphan_size = 0;
      attr_t immigrant_index = 0;
      for (auto &&immigrant :
           sibling->children_ | views::drop(ssize(sibling->children_) - n)) {
        immigrant->parent_ = node;
        immigrant->index_ = immigrant_index++;
        orphan_size += immigrant->size_;
      }
      node->size_ += orphan_size;
      sibling->size_ -= orphan_size;

      ranges::move(sibling->children_ |
                       views::drop(ssize(sibling->children_) - n),
                   back_inserter(node->children_));
      ranges::rotate(node->children_ | views::reverse,
                     node->children_.rbegin() + n);
      for (attr_t idx = 0; idx < n; ++idx) {
        sibling->children_.pop_back();
      }
      attr_t child_index = n;
      for (auto &&child : node->children_ | views::drop(n)) {
        child->index_ = child_index++;
      }
    }
  }

  auto get_lb(const K &key, const Node *x) const noexcept {
    if constexpr (use_linsearch_) {
      auto lbcomp = [&key](const K &other) { return Comp{}(other, key); };
      return distance(x->keys_.begin(),
                      ranges::find_if_not(x->keys_.begin(),
                                          x->keys_.begin() + x->nkeys(), lbcomp,
                                          Proj{}));
    } else {
      return distance(x->keys_.begin(),
                      ranges::lower_bound(x->keys_.begin(),
                                          x->keys_.begin() + x->nkeys(), key,
                                          Comp{}, Proj{}));
    }
  }

  auto get_ub(const K &key, const Node *x) const noexcept {
    if constexpr (use_linsearch_) {
      auto ubcomp = [&key](const K &other) { return Comp{}(key, other); };
      return distance(x->keys_.begin(),
                      ranges::find_if(x->keys_.begin(),
                                      x->keys_.begin() + x->nkeys(), ubcomp,
                                      Proj{}));
    } else {
      return distance(x->keys_.begin(),
                      ranges::upper_bound(x->keys_.begin(),
                                          x->keys_.begin() + x->nkeys(), key,
                                          Comp{}, Proj{}));
    }
  }

  const_iterator_type search(const K &key) const {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) {
        return const_iterator_type(x, static_cast<attr_t>(i));
      } else if (x->is_leaf()) {
        return cend();
      } else {
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  nonconst_iterator_type find_lower_bound(const K &key, bool climb = true) {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (x->is_leaf()) {
        auto it = nonconst_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return nonconst_iterator_type(end());
  }

  const_iterator_type find_lower_bound(const K &key, bool climb = true) const {
    auto x = root_.get();
    while (x) {
      auto i = get_lb(key, x);
      if (x->is_leaf()) {
        auto it = const_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  nonconst_iterator_type find_upper_bound(const K &key, bool climb = true) {
    auto x = root_.get();
    while (x) {
      auto i = get_ub(key, x);
      if (x->is_leaf()) {
        auto it = nonconst_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return nonconst_iterator_type(end());
  }

  const_iterator_type find_upper_bound(const K &key, bool climb = true) const {
    auto x = root_.get();
    while (x) {
      auto i = get_ub(key, x);
      if (x->is_leaf()) {
        auto it = const_iterator_type(x, static_cast<attr_t>(i));
        if (climb) {
          it.climb();
        }
        return it;
      } else {
        x = x->children_[i].get();
      }
    }
    return cend();
  }

  void split_child(Node *y) {
    auto i = y->index_;
    Node *x = y->parent_;

    auto z = make_node();
    z->parent_ = x;
    z->index_ = i + 1;
    z->height_ = y->height_;

    ranges::move(y->keys_ | views::drop(Fanout), back_inserter(z->keys_));

    auto z_size = z->nkeys();
    if (!y->is_leaf()) {
      z->children_.reserve(2 * Fanout);
      ranges::move(y->children_ | views::drop(Fanout),
                   back_inserter(z->children_));
      for (auto &&child : z->children_) {
        child->parent_ = z.get();
        child->index_ -= Fanout;
        z_size += child->size_;
      }
      while (static_cast<attr_t>(ssize(y->children_)) > Fanout) {
        y->children_.pop_back();
      }
    }
    z->size_ = z_size;
    y->size_ -= (z_size + 1);

    x->children_.insert(x->children_.begin() + i + 1, move(z));
    for (auto &&child : x->children_ | views::drop(i + 2)) {
      child->index_++;
    }

    x->keys_.insert(x->keys_.begin() + i, move(y->keys_[Fanout - 1]));
    y->keys_.resize(Fanout - 1);
  }

  void merge_child(Node *y) {
    auto i = y->index_;
    Node *x = y->parent_;
    auto sibling = x->children_[i + 1].get();

    auto immigrated_size = sibling->nkeys();

    y->keys_.push_back(move(x->keys_[i]));
    ranges::move(sibling->keys_, back_inserter(y->keys_));

    if (!y->is_leaf()) {
      attr_t immigrant_index = static_cast<attr_t>(ssize(y->children_));
      for (auto &&child : sibling->children_) {
        child->parent_ = y;
        child->index_ = immigrant_index++;
        immigrated_size += child->size_;
      }
      ranges::move(sibling->children_, back_inserter(y->children_));
    }
    y->size_ += immigrated_size + 1;

    shift_left(x->children_.begin() + i + 1, x->children_.end(), 1);
    x->children_.pop_back();
    shift_left(x->keys_.begin() + i, x->keys_.end(), 1);
    x->keys_.pop_back();

    for (auto &&child : x->children_ | views::drop(i + 1)) {
      child->index_--;
    }
  }

  void try_merge(Node *x, bool left_side) {
    if (ssize(x->children_) < 2) {
      return;
    }
    if (left_side) {
      auto first = x->children_[0].get();
      auto second = x->children_[1].get();

      if (first->nkeys() + second->nkeys() <= 2 * Fanout - 2) {
        merge_child(first);
      } else if (first->nkeys() < Fanout - 1) {
        auto deficit = (Fanout - 1 - first->nkeys());
        left_rotate_n(first, deficit);
      }
    } else {
      auto rfirst = x->children_.back().get();
      auto rsecond = x->children_[ssize(x->children_) - 2].get();

      if (rfirst->nkeys() + rsecond->nkeys() <= 2 * Fanout - 2) {
        merge_child(rsecond);
      } else if (rfirst->nkeys() < Fanout - 1) {
        auto deficit = (Fanout - 1 - rfirst->nkeys());
        right_rotate_n(rfirst, deficit);
      }
    }
  }

  template <typename T>
  iterator_type
  insert_leaf(Node *node, attr_t i,
              T &&value) requires is_same_v<remove_cvref_t<T>, V> {
    bool update_begin = (empty() || Comp{}(Proj{}(value), Proj{}(*begin_)));
    node->keys_.insert(node->keys_.begin() + i, forward<T>(value));
    iterator_type iter(node, i);
    if (update_begin) {
      begin_ = iter;
    }

    auto curr = node;
    while (curr) {
      curr->size_++;
      curr = curr->parent_;
    }

    return iter;
  }

  template <typename T>
  iterator_type
  insert_ub(T &&key) requires(AllowDup &&is_same_v<remove_cvref_t<T>, V>) {
    auto x = root_.get();
    while (true) {
      auto i = get_ub(Proj{}(key), x);
      if (x->is_leaf()) {
        return insert_leaf(x, static_cast<attr_t>(i), forward<T>(key));
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (Comp{}(Proj{}(x->keys_[i]), Proj{}(key))) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  template <typename T>
  pair<iterator_type, bool>
  insert_lb(T &&key) requires(!AllowDup && is_same_v<remove_cvref_t<T>, V>) {
    auto x = root_.get();
    while (true) {
      auto i = get_lb(Proj{}(key), x);
      if (i < x->nkeys() && Proj{}(key) == Proj{}(x->keys_[i])) {
        return {iterator_type(x, static_cast<attr_t>(i)), false};
      } else if (x->is_leaf()) {
        return {insert_leaf(x, static_cast<attr_t>(i), forward<T>(key)), true};
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (Proj{}(key) == Proj{}(x->keys_[i])) {
            return {iterator_type(x, static_cast<attr_t>(i)), false};
          } else if (Comp{}(Proj{}(x->keys_[i]), Proj{}(key))) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  iterator_type erase_leaf(Node *node, attr_t i) {
    bool update_begin = (begin_ == const_iterator_type(node, i));
    shift_left(node->keys_.begin() + i, node->keys_.end(), 1);
    node->keys_.pop_back();
    iterator_type iter(node, i);
    iter.climb();
    if (update_begin) {
      begin_ = iter;
    }
    auto curr = node;
    while (curr) {
      curr->size_--;
      curr = curr->parent_;
    }
    return iter;
  }

  size_t erase_lb(Node *x, const K &key) requires(!AllowDup) {
    while (true) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) {
        if (x->is_leaf()) {
          erase_leaf(x, static_cast<attr_t>(i));
          return 1;
        } else if (x->children_[i]->can_take_key()) {
          nonconst_iterator_type iter(x, static_cast<attr_t>(i));
          auto pred = prev(iter);
          iter_swap(pred, iter);
          x = x->children_[i].get();
        } else if (x->children_[i + 1]->can_take_key()) {
          nonconst_iterator_type iter(x, static_cast<attr_t>(i));
          auto succ = next(iter);
          iter_swap(succ, iter);
          x = x->children_[i + 1].get();
        } else {
          auto next = x->children_[i].get();
          merge_child(next);
          promote_root_if_necessary();
          x = next;
        }
      } else if (x->is_leaf()) {
        return 0;
      } else {
        auto next = x->children_[i].get();
        if (x->children_[i]->has_minimal_keys()) {
          if (i + 1 < ssize(x->children_) &&
              x->children_[i + 1]->can_take_key()) {
            left_rotate(next);
          } else if (i - 1 >= 0 && x->children_[i - 1]->can_take_key()) {
            right_rotate(next);
          } else if (i + 1 < ssize(x->children_)) {
            merge_child(next);
            promote_root_if_necessary();
          } else if (i - 1 >= 0) {
            next = x->children_[i - 1].get();
            merge_child(next);
            promote_root_if_necessary();
          }
        }
        x = next;
      }
    }
  }

  iterator_type erase_hint([[maybe_unused]] const V &value,
                           vector<attr_t> &hints) {
    auto x = root_.get();
    while (true) {
      auto i = hints.back();
      hints.pop_back();
      if (hints.empty()) {
        if (x->is_leaf()) {
          return erase_leaf(x, i);
        } else if (x->children_[i]->can_take_key()) {
          nonconst_iterator_type iter(x, i);
          auto pred = prev(iter);
          iter_swap(pred, iter);
          x = x->children_[i].get();
          auto curr = x;
          while (!curr->is_leaf()) {
            hints.push_back(static_cast<attr_t>(ssize(curr->children_)) - 1);
            curr = curr->children_.back().get();
          }
          hints.push_back(curr->nkeys() - 1);
          ranges::reverse(hints);
        } else if (x->children_[i + 1]->can_take_key()) {
          nonconst_iterator_type iter(x, i);
          auto succ = next(iter);
          iter_swap(succ, iter);
          x = x->children_[i + 1].get();
          auto curr = x;
          while (!curr->is_leaf()) {
            hints.push_back(0);
            curr = curr->children_.front().get();
          }
          hints.push_back(0);
        } else {
          auto next = x->children_[i].get();
          merge_child(next);
          promote_root_if_necessary();
          x = next;
          hints.push_back(Fanout - 1);
        }
      } else {
        auto next = x->children_[i].get();
        if (x->children_[i]->has_minimal_keys()) {
          if (i + 1 < ssize(x->children_) &&
              x->children_[i + 1]->can_take_key()) {
            left_rotate(x->children_[i].get());
          } else if (i - 1 >= 0 && x->children_[i - 1]->can_take_key()) {
            right_rotate(x->children_[i].get());
            hints.back() += 1;
          } else if (i + 1 < ssize(x->children_)) {
            merge_child(next);
            promote_root_if_necessary();
          } else if (i - 1 >= 0) {
            next = x->children_[i - 1].get();
            merge_child(next);
            promote_root_if_necessary();
            hints.back() += Fanout;
          }
        }
        x = next;
      }
    }
  }

private:
  static constexpr attr_t bulk_erase_threshold = 30;

protected:
  size_type erase_range(const_iterator_type first, const_iterator_type last) {
    if (first == cend()) {
      return 0;
    }
    if (first == begin_ && last == cend()) {
      auto cnt = size();
      clear();
      return cnt;
    }

    attr_t first_order = get_order(first);
    attr_t last_order = (last == cend()) ? root_->size_ : get_order(last);

    attr_t cnt = last_order - first_order;
    if (cnt < bulk_erase_threshold) {
      first.climb();
      for (attr_t i = 0; i < cnt; ++i) {
        first = erase(first);
      }
      return cnt;
    }

    K first_key = Proj{}(*first);

    auto [tree1, tree2] = split_to_two_trees(first, last);
    auto final_tree = join(move(tree1), first_key, move(tree2));
    final_tree.erase(final_tree.lower_bound(first_key));

    this->swap(final_tree);
    return cnt;
  }

  V get_kth(attr_t idx) const {
    auto x = root_.get();
    while (x) {
      if (x->is_leaf()) {
        return x->keys_[idx];
      } else {
        attr_t i = 0;
        const auto n = x->nkeys();
        Node *next = nullptr;
        for (; i < n; ++i) {
          auto child_sz = x->children_[i]->size_;
          if (idx < child_sz) {
            next = x->children_[i].get();
            break;
          } else if (idx == child_sz) {
            return x->keys_[i];
          } else {
            idx -= child_sz + 1;
          }
        }
        if (i == n) {
          next = x->children_[n].get();
        }
        x = next;
      }
    }
    throw runtime_error("unreachable");
  }

  attr_t get_order(const_iterator_type iter) const {
    auto [node, idx] = iter;
    attr_t order = 0;
    if (!node->is_leaf()) {
      for (attr_t i = 0; i <= idx; ++i) {
        order += node->children_[i]->size_;
      }
    }
    order += idx;
    while (node->parent_) {
      for (attr_t i = 0; i < node->index_; ++i) {
        order += node->parent_->children_[i]->size_;
      }
      order += node->index_;
      node = node->parent_;
    }
    return order;
  }

public:
  iterator_type find(const K &key) { return iterator_type(search(key)); }

  const_iterator_type find(const K &key) const { return search(key); }

  bool contains(const K &key) const { return search(key) != cend(); }

  iterator_type lower_bound(const K &key) {
    return iterator_type(find_lower_bound(key));
  }

  const_iterator_type lower_bound(const K &key) const {
    return const_iterator_type(find_lower_bound(key));
  }

  iterator_type upper_bound(const K &key) {
    return iterator_type(find_upper_bound(key));
  }

  const_iterator_type upper_bound(const K &key) const {
    return const_iterator_type(find_upper_bound(key));
  }

  ranges::subrange<iterator_type> equal_range(const K &key) {
    return {iterator_type(find_lower_bound(key)),
            iterator_type(find_upper_bound(key))};
  }

  ranges::subrange<const_iterator_type> equal_range(const K &key) const {
    return {const_iterator_type(find_lower_bound(key)),
            const_iterator_type(find_upper_bound(key))};
  }

  ranges::subrange<const_iterator_type> enumerate(const K &a,
                                                  const K &b) const {
    if (Comp{}(b, a)) {
      throw invalid_argument("b < a in enumerate()");
    }
    return {const_iterator_type(find_lower_bound(a)),
            const_iterator_type(find_upper_bound(b))};
  }

  V kth(attr_t idx) const {
    if (idx >= root_->size_) {
      throw invalid_argument("in kth() k >= size()");
    }
    return get_kth(idx);
  }

  attr_t order(const_iterator_type iter) const {
    if (iter == cend()) {
      throw invalid_argument("attempt to get order in end()");
    }
    return get_order(iter);
  }

  attr_t count(const K &key) const requires(AllowDup) {
    auto first = find_lower_bound(key);
    auto last = find_upper_bound(key);
    attr_t first_order = get_order(first);
    attr_t last_order = (last == cend()) ? root_->size_ : get_order(last);
    return last_order - first_order;
  }

protected:
  template <typename T>
  conditional_t<AllowDup, iterator_type, pair<iterator_type, bool>>
  insert_value(T &&key) requires(is_same_v<remove_cvref_t<T>, V>) {
    if (root_->is_full()) {
      auto new_root = make_node();
      root_->parent_ = new_root.get();
      new_root->size_ = root_->size_;
      new_root->height_ = root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(move(root_));
      root_ = move(new_root);

      split_child(root_->children_[0].get());
    }
    if constexpr (AllowDup) {
      return insert_ub(forward<T>(key));
    } else {
      return insert_lb(forward<T>(key));
    }
  }

  vector<attr_t> get_path_from_root(const_iterator_type iter) const {
    auto node = iter.node_;
    vector<attr_t> hints;
    hints.push_back(iter.index_);
    while (node && node->parent_) {
      hints.push_back(node->index_);
      node = node->parent_;
    }
    return hints;
  }

public:
  conditional_t<AllowDup, iterator_type, pair<iterator_type, bool>>
  insert(const V &key) {
    return insert_value(key);
  }

  conditional_t<AllowDup, iterator_type, pair<iterator_type, bool>>
  insert(V &&key) {
    return insert_value(move(key));
  }

  template <typename... Args>
  conditional_t<AllowDup, iterator_type, pair<iterator_type, bool>>
  emplace(Args &&... args) requires is_constructible_v<V, Args...> {
    V val{forward<Args>(args)...};
    return insert_value(move(val));
  }

  template <typename T>
  auto &operator[](T &&raw_key) requires(!is_set_ && !AllowDup) {
    if (root_->is_full()) {
      auto new_root = make_node();
      root_->parent_ = new_root.get();
      new_root->size_ = root_->size_;
      new_root->height_ = root_->height_ + 1;
      new_root->children_.reserve(Fanout * 2);
      new_root->children_.push_back(move(root_));
      root_ = move(new_root);

      split_child(root_->children_[0].get());
    }

    K key{forward<T>(raw_key)};
    auto x = root_.get();
    while (true) {
      auto i = get_lb(key, x);
      if (i < x->nkeys() && key == Proj{}(x->keys_[i])) {
        return iterator_type(x, static_cast<attr_t>(i))->second;
      } else if (x->is_leaf()) {
        V val{move(key), {}};
        return insert_leaf(x, static_cast<attr_t>(i), move(val))->second;
      } else {
        if (x->children_[i]->is_full()) {
          split_child(x->children_[i].get());
          if (key == Proj{}(x->keys_[i])) {
            return iterator_type(x, static_cast<attr_t>(i))->second;
          } else if (Comp{}(Proj{}(x->keys_[i]), key)) {
            ++i;
          }
        }
        x = x->children_[i].get();
      }
    }
  }

  template <forward_iterator Iter>
  requires is_constructible_v<V, iter_reference_t<Iter>>
      size_type insert_range(Iter first, Iter last) {
    auto [min_elem, max_elem] =
        ranges::minmax_element(first, last, Comp{}, Proj{});
    auto lb = find_lower_bound(*min_elem);
    auto ub = find_upper_bound(*max_elem);
    if (lb != ub) {
      size_type sz = 0;
      for (; first != last; ++first) {
        if constexpr (AllowDup) {
          insert(*first);
          sz++;
        } else {
          auto [_, inserted] = insert(*first);
          if (inserted) {
            sz++;
          }
        }
      }
      return sz;
    } else {
      BTreeBase tree_mid;
      for (; first != last; ++first) {
        tree_mid.insert(*first);
      }
      auto sz = tree_mid.size();
      auto [tree_left, tree_right] = split_to_two_trees(lb, ub);
      auto tree_leftmid = join(move(tree_left), move(tree_mid));
      auto final_tree = join(move(tree_leftmid), move(tree_right));
      this->swap(final_tree);
      return sz;
    }
  }

  template <ranges::forward_range R>
  requires is_constructible_v<V, ranges::range_reference_t<R>>
      size_type insert_range(R &&r) {
    return insert_range(r.begin(), r.end());
  }

  const_iterator_type erase(const_iterator_type iter) {
    if (iter == cend()) {
      throw invalid_argument("attempt to erase cend()");
    }
    vector<attr_t> hints = get_path_from_root(iter);
    V value(move(*iter));
    return erase_hint(value, hints);
  }

  size_type erase(const K &key) {
    if constexpr (AllowDup) {
      return erase_range(const_iterator_type(find_lower_bound(key, false)),
                         const_iterator_type(find_upper_bound(key, false)));
    } else {
      return erase_lb(root_.get(), key);
    }
  }

  size_type erase_range(const K &a, const K &b) {
    if (Comp{}(b, a)) {
      throw invalid_argument("b < a in erase_range()");
    }
    return erase_range(const_iterator_type(find_lower_bound(a)),
                       const_iterator_type(find_upper_bound(b)));
  }

  template <typename Pred> size_type erase_if(Pred pred) {
    auto old_size = size();
    auto it = begin_;
    for (; it != end();) {
      if (pred(*it)) {
        it = erase(it);
      } else {
        ++it;
      }
    }
    return old_size - size();
  }

public:
  template <typename K_, typename V_, typename Comp_, bool AllowDup_>
  friend struct join_helper;

protected:
  pair<BTreeBase, BTreeBase> split_to_two_trees(const_iterator_type iter_lb,
                                                const_iterator_type iter_ub) {
    BTreeBase tree_left;
    BTreeBase tree_right;
    auto lindices = get_path_from_root(iter_lb);
    auto xl = iter_lb.node_;
    ranges::reverse(lindices);
    auto rindices = get_path_from_root(iter_ub);
    auto xr = iter_ub.node_;
    ranges::reverse(rindices);

    while (!lindices.empty()) {
      auto il = lindices.back();
      lindices.pop_back();

      auto lroot = tree_left.root_.get();

      if (xl->is_leaf()) {
        if (il > 0) {
          ranges::move(xl->keys_ | views::take(il),
                       back_inserter(lroot->keys_));
          lroot->size_ += il;
        }

        xl = xl->parent_;
      } else {
        if (il > 0) {
          BTreeBase supertree_left;
          auto slroot = supertree_left.root_.get();
          ranges::move(xl->keys_ | views::take(il - 1),
                       back_inserter(slroot->keys_));
          slroot->size_ += (il - 1);
          slroot->children_.reserve(Fanout * 2);

          ranges::move(xl->children_ | views::take(il),
                       back_inserter(slroot->children_));
          slroot->height_ = slroot->children_[0]->height_ + 1;
          for (auto &&sl_child : slroot->children_) {
            sl_child->parent_ = slroot;
            slroot->size_ += sl_child->size_;
          }

          supertree_left.promote_root_if_necessary();
          supertree_left.set_begin();

          BTreeBase new_tree_left = join(
              move(supertree_left), move(xl->keys_[il - 1]), move(tree_left));
          tree_left = move(new_tree_left);
        }

        xl = xl->parent_;
      }
    }
    while (!rindices.empty()) {
      auto ir = rindices.back();
      rindices.pop_back();

      auto rroot = tree_right.root_.get();

      if (xr->is_leaf()) {
        assert(rroot->size_ == 0);

        if (ir < xr->nkeys()) {
          auto immigrants = xr->nkeys() - ir;
          ranges::move(xr->keys_ | views::drop(ir),
                       back_inserter(rroot->keys_));
          rroot->size_ += immigrants;
        }

        xr = xr->parent_;
      } else {

        if (ir + 1 < ssize(xr->children_)) {
          BTreeBase supertree_right;
          auto srroot = supertree_right.root_.get();

          auto immigrants = xr->nkeys() - (ir + 1);
          ranges::move(xr->keys_ | views::drop(ir + 1),
                       back_inserter(srroot->keys_));
          srroot->size_ += immigrants;

          srroot->children_.reserve(Fanout * 2);

          ranges::move(xr->children_ | views::drop(ir + 1),
                       back_inserter(srroot->children_));
          srroot->height_ = srroot->children_[0]->height_ + 1;
          attr_t sr_index = 0;
          for (auto &&sr_child : srroot->children_) {
            sr_child->parent_ = srroot;
            sr_child->index_ = sr_index++;
            srroot->size_ += sr_child->size_;
          }

          supertree_right.promote_root_if_necessary();
          supertree_right.set_begin();

          BTreeBase new_tree_right = join(move(tree_right), move(xr->keys_[ir]),
                                          move(supertree_right));
          tree_right = move(new_tree_right);
        }

        xr = xr->parent_;
      }
    }
    clear();
    return {move(tree_left), move(tree_right)};
  }

public:
  template <typename K_, typename V_, typename Comp_, bool AllowDup_,
            typename T>
  friend struct split_helper;
};

template <typename K, typename V, typename Comp, bool AllowDup>
struct join_helper {
  BTreeBase<K, V, Comp, AllowDup> result_;

private:
  using Tree = BTreeBase<K, V, Comp, AllowDup>;
  using Node = typename Tree::node_type;
  using Proj = typename Tree::Proj;

public:
  join_helper(BTreeBase<K, V, Comp, AllowDup> &&tree_left,
              BTreeBase<K, V, Comp, AllowDup> &&tree_right) {
    if (tree_left.empty()) {
      result_ = move(tree_right);
    } else if (tree_right.empty()) {
      result_ = move(tree_left);
    } else {
      auto it = tree_right.begin();
      V mid_value = *it;
      tree_right.erase(it);
      result_ = join(move(tree_left), move(mid_value), move(tree_right));
    }
  }

  template <typename T_>
  join_helper(BTreeBase<K, V, Comp, AllowDup> &&tree_left, T_ &&raw_value,
              BTreeBase<K, V, Comp, AllowDup> &&tree_right) {

    V mid_value{forward<T_>(raw_value)};
    if ((!tree_left.empty() &&
         Comp{}(Proj{}(mid_value), Proj{}(*tree_left.crbegin()))) ||
        (!tree_right.empty() &&
         Comp{}(Proj{}(*tree_right.cbegin()), Proj{}(mid_value)))) {
      throw invalid_argument("Join() key order is invalid\n");
    }

    auto height_left = tree_left.root_->height_;
    auto height_right = tree_right.root_->height_;
    auto size_left = tree_left.root_->size_;
    auto size_right = tree_right.root_->size_;

    if (height_left >= height_right) {
      Tree new_tree = move(tree_left);
      attr_t curr_height = height_left;
      Node *curr = new_tree.root_.get();
      if (new_tree.root_->is_full()) {
        auto new_root = new_tree.make_node();
        new_tree.root_->index_ = 0;
        new_tree.root_->parent_ = new_root.get();
        new_root->size_ = new_tree.root_->size_;
        new_root->height_ = new_tree.root_->height_ + 1;
        new_root->children_.reserve(Fanout * 2);
        new_root->children_.push_back(move(new_tree.root_));
        new_tree.root_ = move(new_root);

        new_tree.split_child(new_tree.root_->children_[0].get());
        curr = new_tree.root_->children_[1].get();
      }

      while (curr && curr_height > height_right) {
        assert(!curr->is_leaf());
        curr_height--;

        if (curr->children_.back()->is_full()) {
          new_tree.split_child(curr->children_.back().get());
        }
        curr = curr->children_.back().get();
      }

      auto parent = curr->parent_;
      if (!parent) {
        auto new_root = tree_left.make_node();
        new_root->height_ = new_tree.root_->height_ + 1;
        new_root->keys_.push_back(move(mid_value));
        new_root->children_.reserve(Fanout * 2);

        new_tree.root_->parent_ = new_root.get();
        new_tree.root_->index_ = 0;
        new_root->children_.push_back(move(new_tree.root_));

        tree_right.root_->parent_ = new_root.get();
        tree_right.root_->index_ = 1;
        new_root->children_.push_back(move(tree_right.root_));

        new_tree.root_ = move(new_root);
        new_tree.try_merge(new_tree.root_.get(), false);
        new_tree.promote_root_if_necessary();
        new_tree.root_->size_ = size_left + size_right + 1;
      } else {
        parent->keys_.push_back(move(mid_value));

        tree_right.root_->parent_ = parent;
        tree_right.root_->index_ =
            static_cast<attr_t>(ssize(parent->children_));
        parent->children_.push_back(move(tree_right.root_));

        while (parent) {
          parent->size_ += (size_right + 1);
          new_tree.try_merge(parent, false);
          parent = parent->parent_;
        }
        new_tree.promote_root_if_necessary();
      }
      result_ = move(new_tree);
    } else {
      Tree new_tree = move(tree_right);
      attr_t curr_height = height_right;
      Node *curr = new_tree.root_.get();
      if (new_tree.root_->is_full()) {
        auto new_root = new_tree.make_node();
        new_tree.root_->index_ = 0;
        new_tree.root_->parent_ = new_root.get();
        new_root->size_ = new_tree.root_->size_;
        new_root->height_ = new_tree.root_->height_ + 1;
        new_root->children_.reserve(Fanout * 2);
        new_root->children_.push_back(move(new_tree.root_));
        new_tree.root_ = move(new_root);

        new_tree.split_child(new_tree.root_->children_[0].get());
        curr = new_tree.root_->children_[0].get();
      }

      while (curr && curr_height > height_left) {
        curr_height--;

        if (curr->children_.front()->is_full()) {
          new_tree.split_child(curr->children_[0].get());
        }
        curr = curr->children_.front().get();
      }
      auto parent = curr->parent_;
      parent->keys_.insert(parent->keys_.begin(), move(mid_value));

      auto new_begin = tree_left.begin();
      tree_left.root_->parent_ = parent;
      tree_left.root_->index_ = 0;
      parent->children_.insert(parent->children_.begin(),
                               move(tree_left.root_));
      for (auto &&child : parent->children_ | views::drop(1)) {
        child->index_++;
      }
      while (parent) {
        parent->size_ += (size_left + 1);
        new_tree.try_merge(parent, true);
        parent = parent->parent_;
      }
      new_tree.promote_root_if_necessary();
      new_tree.begin_ = new_begin;
      result_ = move(new_tree);
    }
  }
  BTreeBase<K, V, Comp, AllowDup> &&result() { return move(result_); }
};
template <typename K, typename V, typename Comp, bool AllowDup, typename T>
struct split_helper {
private:
  pair<BTreeBase<K, V, Comp, AllowDup>, BTreeBase<K, V, Comp, AllowDup>>
      result_;

public:
  using Tree = BTreeBase<K, V, Comp, AllowDup>;

  split_helper(BTreeBase<K, V, Comp, AllowDup> &&tree,
               T &&raw_key) requires(is_constructible_v<K, remove_cvref_t<T>>) {
    if (tree.empty()) {
      Tree tree_left, tree_right;
      result_ = {move(tree_left), move(tree_right)};
    } else {
      K mid_key{forward<T>(raw_key)};
      result_ = tree.split_to_two_trees(tree.find_lower_bound(mid_key, false),
                                        tree.find_upper_bound(mid_key, false));
    }
  }
  split_helper(
      BTreeBase<K, V, Comp, AllowDup> &&tree, T &&raw_key1,
      T &&raw_key2) requires(is_constructible_v<K, remove_cvref_t<T>>) {
    if (tree.empty()) {
      Tree tree_left, tree_right;
      result_ = {move(tree_left), move(tree_right)};
    } else {
      K key1{forward<T>(raw_key1)};
      K key2{forward<T>(raw_key2)};
      if (Comp{}(key2, key1)) {
        throw invalid_argument("split() key order is invalid\n");
      }
      result_ = tree.split_to_two_trees(tree.find_lower_bound(key1, false),
                                        tree.find_upper_bound(key2, false));
    }
  }
  pair<BTreeBase<K, V, Comp, AllowDup>, BTreeBase<K, V, Comp, AllowDup>> &&
  result() {
    return move(result_);
  }
};

template <typename K, typename Comp = ranges::less>
using BTreeSet = BTreeBase<K, K, Comp, false>;

template <typename K, typename Comp = ranges::less>
using BTreeMultiSet = BTreeBase<K, K, Comp, true>;

template <typename K, typename V, typename Comp = ranges::less>
using BTreeMap = BTreeBase<K, BTreePair<K, V>, Comp, false>;

template <typename K, typename V, typename Comp = ranges::less>
using BTreeMultiMap = BTreeBase<K, BTreePair<K, V>, Comp, true>;

template <typename K, typename V, typename Comp, bool AllowDup>
BTreeBase<K, V, Comp, AllowDup>
join(BTreeBase<K, V, Comp, AllowDup> &&tree_left,
     BTreeBase<K, V, Comp, AllowDup> &&tree_right) {
  return join_helper(move(tree_left), move(tree_right)).result();
}

template <typename K, typename V, typename Comp, bool AllowDup, typename T_>
BTreeBase<K, V, Comp, AllowDup>
join(BTreeBase<K, V, Comp, AllowDup> &&tree_left, T_ &&raw_value,
     BTreeBase<K, V, Comp, AllowDup> &&tree_right) {
  return join_helper(move(tree_left), move(raw_value), move(tree_right))
      .result();
}

template <typename K, typename V, typename Comp, bool AllowDup, typename T>
pair<BTreeBase<K, V, Comp, AllowDup>, BTreeBase<K, V, Comp, AllowDup>>
split(BTreeBase<K, V, Comp, AllowDup> &&tree, T &&raw_key) {
  return split_helper(move(tree), move(raw_key)).result();
}

template <typename K, typename V, typename Comp, bool AllowDup, typename T>
pair<BTreeBase<K, V, Comp, AllowDup>, BTreeBase<K, V, Comp, AllowDup>>
split(BTreeBase<K, V, Comp, AllowDup> &&tree, T &&raw_key1, T &&raw_key2) {
  return split_helper(move(tree), move(raw_key1), move(raw_key2)).result();
}
} // namespace frozenca

#endif //__FC_BTREE_H__
