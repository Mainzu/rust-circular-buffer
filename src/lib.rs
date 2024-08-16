#![cfg_attr(not(test), no_std)]

use core::mem::MaybeUninit;
use core::ops::{Index, IndexMut};

/// A circular buffer implementation with a fixed capacity.
///
/// This structure provides O(1) enqueue and dequeue operations, making it
/// efficient for scenarios where a fixed-size buffer with fast insertion
/// and removal at both ends is needed.
///
/// The capacity of the buffer is fixed, a power of 2 allow for optimal performance.
///
/// # Examples
///
/// ```
/// use circular_buffer::CircularBuffer;
///
/// let mut buffer = CircularBuffer::<i32, 3>::new();
/// buffer.enqueue(1);
/// buffer.enqueue(2);
/// assert_eq!(buffer.len(), 2);
/// assert_eq!(buffer.dequeue(), Some(1));
/// ```
#[derive(Debug)]
pub struct CircularBuffer<T, const N: usize> {
    buffer: [MaybeUninit<T>; N],
    head: usize,
    size: usize,
}

impl<T, const N: usize> CircularBuffer<T, N> {
    const BIT_COUNT: usize = Self::count_bits(N);
    const BIT_MASK: usize = if N == 0 { 0 } else { N - 1 };

    // Private helper methods

    /// Counts the number of set bits in a usize value.
    const fn count_bits(mut v: usize) -> usize {
        let mut c = 0;
        while v != 0 {
            v &= v - 1;
            c += 1;
        }
        c
    }

    /// Calculates the actual index in the buffer array for a given logical index.
    fn get_index(&self, idx: usize) -> usize {
        if Self::BIT_COUNT == 0 || Self::BIT_COUNT == 1 {
            (self.head + idx) & Self::BIT_MASK
        } else {
            (self.head + idx) % N
        }
    }

    // Public methods

    /// Creates a new, empty `CircularBuffer`.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let buffer = CircularBuffer::<i32, 8>::new();
    /// assert!(buffer.is_empty());
    /// ```
    pub const fn new() -> Self {
        Self {
            buffer: unsafe { MaybeUninit::uninit().assume_init() },
            head: 0,
            size: 0,
        }
    }

    /// Returns the capacity of the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let buffer = CircularBuffer::<i32, 8>::new();
    /// assert_eq!(buffer.capacity(), 8);
    /// ```
    pub const fn capacity(&self) -> usize {
        N
    }

    /// Returns the number of elements in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 8>::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// assert_eq!(buffer.len(), 2);
    /// ```
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Returns `true` if the buffer contains no elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 8>::new();
    /// assert!(buffer.is_empty());
    /// buffer.enqueue(1);
    /// assert!(!buffer.is_empty());
    /// ```
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns `true` if the buffer is at full capacity.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 8>::new();
    /// assert!(!buffer.is_full());
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// assert!(buffer.is_full());
    /// ```
    pub const fn is_full(&self) -> bool {
        self.size == N
    }

    /// Adds an element to the back of the buffer.
    ///
    /// If the buffer is full, the oldest element is overwritten and returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 2>::new();
    /// assert_eq!(buffer.enqueue(1), None);
    /// assert_eq!(buffer.enqueue(2), None);
    /// assert_eq!(buffer.enqueue(3), Some(1)); // 1 is overwritten and returned
    /// ```
    pub fn enqueue(&mut self, value: T) -> Option<T> {
        if self.is_full() {
            let old_value = unsafe { self.buffer[self.head].assume_init_read() };
            self.buffer[self.head] = MaybeUninit::new(value);
            self.head = self.get_index(1);
            Some(old_value)
        } else {
            let idx = self.get_index(self.size);
            self.buffer[idx] = MaybeUninit::new(value);
            self.size += 1;
            None
        }
    }

    /// Adds an element to the back of the buffer if it is not full.
    ///
    /// Returns the original value was if the buffer was full.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 2>::new();
    /// assert!(buffer.try_enqueue(1).is_ok());
    /// assert!(buffer.try_enqueue(2).is_ok());
    /// assert!(buffer.try_enqueue(3).is_err());
    /// ```
    pub fn try_enqueue(&mut self, value: T) -> Result<(), T> {
        if self.is_full() {
            Err(value)
        } else {
            let idx = self.get_index(self.size);
            self.buffer[idx] = MaybeUninit::new(value);
            self.size += 1;
            Ok(())
        }
    }

    /// Removes and returns the oldest element from the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer = CircularBuffer::<i32, 2>::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// assert_eq!(buffer.dequeue(), Some(1));
    /// assert_eq!(buffer.dequeue(), Some(2));
    /// assert_eq!(buffer.dequeue(), None);
    /// ```
    pub fn dequeue(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let value = unsafe { self.buffer[self.head].assume_init_read() };
            self.head = self.get_index(1);
            self.size -= 1;
            Some(value)
        }
    }

    /// Returns a reference to the first (oldest) element in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer: CircularBuffer<i32, 2> = CircularBuffer::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// assert_eq!(buffer.first(), Some(&1));
    /// ```
    pub fn first(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { self.buffer[self.head].assume_init_ref() })
        }
    }

    /// Returns a mutable reference to the first (oldest) element in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer: CircularBuffer<i32, 2> = CircularBuffer::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// if let Some(first) = buffer.first_mut() {
    ///     *first = 3;
    /// }
    /// assert_eq!(buffer.first(), Some(&3));
    /// ```
    pub fn first_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            None
        } else {
            Some(unsafe { self.buffer[self.head].assume_init_mut() })
        }
    }

    /// Returns a reference to the last (newest) element in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer: CircularBuffer<i32, 2> = CircularBuffer::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// assert_eq!(buffer.last(), Some(&2));
    /// ```
    pub fn last(&self) -> Option<&T> {
        if self.is_empty() {
            None
        } else {
            let idx = self.get_index(self.size - 1);
            Some(unsafe { self.buffer[idx].assume_init_ref() })
        }
    }

    /// Returns a mutable reference to the last (newest) element in the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer: CircularBuffer<i32, 2> = CircularBuffer::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// if let Some(last) = buffer.last_mut() {
    ///     *last = 3;
    /// }
    /// assert_eq!(buffer.last(), Some(&3));
    /// ```
    pub fn last_mut(&mut self) -> Option<&mut T> {
        if self.is_empty() {
            None
        } else {
            let idx = self.get_index(self.size - 1);
            Some(unsafe { self.buffer[idx].assume_init_mut() })
        }
    }

    /// Removes all elements from the buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use circular_buffer::CircularBuffer;
    ///
    /// let mut buffer: CircularBuffer<i32, 2> = CircularBuffer::new();
    /// buffer.enqueue(1);
    /// buffer.enqueue(2);
    /// buffer.clear();
    /// assert!(buffer.is_empty());
    /// ```
    pub fn clear(&mut self) {
        while let Some(_) = self.dequeue() {}
    }
}

impl<T, const N: usize> Default for CircularBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> Index<usize> for CircularBuffer<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.size, "Index out of bounds");
        let idx = self.get_index(index);
        unsafe { self.buffer[idx].assume_init_ref() }
    }
}

impl<T, const N: usize> IndexMut<usize> for CircularBuffer<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.size, "Index out of bounds");
        let idx = self.get_index(index);
        unsafe { self.buffer[idx].assume_init_mut() }
    }
}

impl<T, const N: usize> Drop for CircularBuffer<T, N> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T: Clone, const N: usize> Clone for CircularBuffer<T, N> {
    fn clone(&self) -> Self {
        let mut new_buffer = Self::new();
        for i in 0..self.size {
            let idx = self.get_index(i);
            new_buffer.enqueue(unsafe { self.buffer[idx].assume_init_ref().clone() });
        }
        new_buffer
    }
}

impl<T, const N: usize> IntoIterator for CircularBuffer<T, N> {
    type Item = T;
    type IntoIter = IntoIter<T, N>;

    fn into_iter(self) -> Self::IntoIter {
        IntoIter { buffer: self }
    }
}

pub struct IntoIter<T, const N: usize> {
    buffer: CircularBuffer<T, N>,
}

impl<T, const N: usize> Iterator for IntoIter<T, N> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.buffer.dequeue()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a CircularBuffer<T, N> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T, N>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            buffer: self,
            index: 0,
        }
    }
}

pub struct Iter<'a, T, const N: usize> {
    buffer: &'a CircularBuffer<T, N>,
    index: usize,
}

impl<'a, T, const N: usize> Iterator for Iter<'a, T, N> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.buffer.len() {
            let item = &self.buffer[self.index];
            self.index += 1;
            Some(item)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test private methods
    #[test]
    fn test_get_index() {
        let buf: CircularBuffer<i32, 4> = CircularBuffer::new();
        assert_eq!(buf.get_index(0), 0);
        assert_eq!(buf.get_index(1), 1);
        assert_eq!(buf.get_index(3), 3);
        assert_eq!(buf.get_index(4), 0);
        assert_eq!(buf.get_index(5), 1);
    }

    // Test public methods
    #[test]
    fn test_new() {
        let buf: CircularBuffer<i32, 5> = CircularBuffer::new();
        assert_eq!(buf.capacity(), 5);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        assert!(!buf.is_full());
    }

    #[test]
    fn test_enqueue() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();

        assert_eq!(buf.enqueue(1), None);
        assert_eq!(buf.len(), 1);

        assert_eq!(buf.enqueue(2), None);
        assert_eq!(buf.len(), 2);

        assert_eq!(buf.enqueue(3), None);
        assert_eq!(buf.len(), 3);
        assert!(buf.is_full());

        assert_eq!(buf.enqueue(4), Some(1));
        assert_eq!(buf.len(), 3);
        assert!(buf.is_full());
    }

    #[test]
    fn test_dequeue() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        buf.enqueue(3);

        assert_eq!(buf.dequeue(), Some(1));
        assert_eq!(buf.len(), 2);

        assert_eq!(buf.dequeue(), Some(2));
        assert_eq!(buf.len(), 1);

        assert_eq!(buf.dequeue(), Some(3));
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());

        assert_eq!(buf.dequeue(), None);
    }

    #[test]
    fn test_front_back() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        assert_eq!(buf.first(), None);
        assert_eq!(buf.last(), None);

        buf.enqueue(1);
        assert_eq!(buf.first(), Some(&1));
        assert_eq!(buf.last(), Some(&1));

        buf.enqueue(2);
        assert_eq!(buf.first(), Some(&1));
        assert_eq!(buf.last(), Some(&2));

        buf.enqueue(3);
        assert_eq!(buf.first(), Some(&1));
        assert_eq!(buf.last(), Some(&3));

        buf.dequeue();
        assert_eq!(buf.first(), Some(&2));
        assert_eq!(buf.last(), Some(&3));
    }

    #[test]
    fn test_indexing() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        buf.enqueue(3);

        assert_eq!(buf[0], 1);
        assert_eq!(buf[1], 2);
        assert_eq!(buf[2], 3);

        buf.dequeue();
        buf.enqueue(4);

        assert_eq!(buf[0], 2);
        assert_eq!(buf[1], 3);
        assert_eq!(buf[2], 4);
    }

    #[test]
    fn test_clear() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        buf.clear();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
    }

    #[test]
    fn test_iteration() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        buf.enqueue(3);

        let mut iter = buf.into_iter();
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_wrapped_buffer() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        buf.enqueue(3);
        buf.dequeue();
        buf.enqueue(4);

        assert_eq!(buf[0], 2);
        assert_eq!(buf[1], 3);
        assert_eq!(buf[2], 4);

        let collected: Vec<i32> = buf.into_iter().collect();
        assert_eq!(collected, vec![2, 3, 4]);
    }

    #[test]
    #[should_panic(expected = "Index out of bounds")]
    fn test_index_out_of_bounds() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);
        let _ = buf[2]; // This should panic
    }

    #[test]
    fn test_clone() {
        let mut buf: CircularBuffer<i32, 3> = CircularBuffer::new();
        buf.enqueue(1);
        buf.enqueue(2);

        let cloned = buf.clone();
        assert_eq!(buf.len(), cloned.len());
        assert_eq!(buf[0], cloned[0]);
        assert_eq!(buf[1], cloned[1]);
    }

    // TODO: test 0 capacity
}
