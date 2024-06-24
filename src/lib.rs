use std::iter::FromIterator;
use std::mem::MaybeUninit;
use std::ops::{Index, IndexMut};

#[inline]
fn count_bits(v: usize) -> usize {
    let mut c = 0;
    let mut v = v;
    while v != 0 {
        v &= v - 1;
        c += 1;
    }
    c
}

struct CircularBuffer<T, const N: usize> {
    buffer: [Option<T>; N],
    head: usize,
    size: usize,
}

impl<T, const N: usize> CircularBuffer<T, N> {
    const BUFFER_SIZE: usize = N;
    const BIT_COUNT: usize = count_bits(N);
    const BIT_MASK: usize = if N == 0 { 0 } else { N - 1 };

    #[track_caller]
    fn bounds_check(&self, idx: usize) {
        if idx >= self.size {
            panic!("Index out of range");
        }
    }

    fn buffer_begin(&self) -> usize {
        self.buffer.as_ptr() as usize
    }

    fn get_index(&self, idx: usize) -> usize {
        if Self::BIT_COUNT == 0 || Self::BIT_COUNT == 1 {
            (self.head - self.buffer_begin() + idx) & Self::BIT_MASK
        } else {
            (self.head - self.buffer_begin() + idx) % Self::BUFFER_SIZE
        }
    }

    fn get_offset(&self, idx: usize) -> isize {
        if self.head + idx >= self.buffer_begin() + N {
            idx as isize - Self::BUFFER_SIZE as isize
        } else {
            idx as isize
        }
    }

    fn get_mut(&mut self, idx: usize) -> &mut T {
        self.buffer
            .get_mut((self.head as isize + self.get_offset(idx)) as usize)
            .unwrap()
            .as_mut()
            .unwrap()
    }

    fn get(&self, idx: usize) -> &T {
        self.buffer
            .get((self.head as isize + self.get_offset(idx)) as usize)
            .unwrap()
            .as_ref()
            .unwrap()
    }
}

impl<T, const N: usize> CircularBuffer<T, N> {
    pub fn new() -> Self {
        Self {
            buffer: std::array::from_fn(|_| None),
            head: std::ptr::null_mut(),
            size: 0,
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        assert!(cap <= N);
        let mut buffer = Self::new();
        buffer.head = buffer.buffer.as_mut_ptr();
        buffer
    }
}

impl<T, const N: usize> FromIterator<T> for CircularBuffer<T, N> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut buffer = Self::with_capacity(N);
        for (i, item) in iter.into_iter().enumerate() {
            if i < N {
                unsafe {
                    std::ptr::write(buffer.get_mut(i), item);
                }
            } else {
                break;
            }
        }
        buffer.size = buffer.buffer.len();
        buffer
    }
}

impl<T, const N: usize> Default for CircularBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T, const N: usize> CircularBuffer<T, N> {
    pub fn capacity(&self) -> usize {
        N
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub fn is_full(&self) -> bool {
        self.size == N
    }

    pub fn data(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.buffer.as_ptr(), self.size) }
    }

    pub fn data_mut(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.buffer.as_mut_ptr(), self.size) }
    }
}

impl<T, const N: usize> Index<usize> for CircularBuffer<T, N> {
    type Output = T;

    fn index(&self, idx: usize) -> &Self::Output {
        self.bounds_check(idx);
        unsafe { &*self.get(idx) }
    }
}

impl<T, const N: usize> IndexMut<usize> for CircularBuffer<T, N> {
    fn index_mut(&mut self, idx: usize) -> &mut Self::Output {
        self.bounds_check(idx);
        unsafe { &mut *self.get_mut(idx) }
    }
}

impl<T, const N: usize> CircularBuffer<T, N> {
    pub fn front(&self) -> &T {
        self.bounds_check(0);
        unsafe { &*self.buffer.as_ptr() }
    }

    pub fn front_mut(&mut self) -> &mut T {
        self.bounds_check(0);
        unsafe { &mut *self.buffer.as_mut_ptr() }
    }

    pub fn back(&self) -> &T {
        let idx = if self.size == 0 { 0 } else { self.size - 1 };
        self.bounds_check(idx);
        unsafe { &*self.get(idx) }
    }

    pub fn back_mut(&mut self) -> &mut T {
        let idx = if self.size == 0 { 0 } else { self.size - 1 };
        self.bounds_check(idx);
        unsafe { &mut *self.get_mut(idx) }
    }

    pub fn enqueue(&mut self, value: T) -> Option<T> {
        if self.is_full() {
            let old_head = unsafe { std::ptr::read(self.buffer.as_ptr()) };
            unsafe {
                std::ptr::write(self.buffer.as_ptr(), value);
            }
            self.head = self.get_mut(1);
            Some(old_head)
        } else {
            let tail = self.get_mut(self.size);
            unsafe {
                std::ptr::write(tail, value);
            }
            self.size += 1;
            None
        }
    }

    pub fn try_enqueue(&mut self, value: T) -> Option<T> {
        if self.is_full() {
            Some(value)
        } else {
            let tail = self.get_mut(self.size);
            unsafe {
                std::ptr::write(tail, value);
            }
            self.size += 1;
            None
        }
    }

    pub fn dequeue(&mut self) -> Option<T> {
        if self.is_empty() {
            None
        } else {
            let head = self.buffer.as_ptr();
            let value = unsafe { std::ptr::read(head) };
            self.head = self.get_mut(1);
            self.size -= 1;
            Some(value)
        }
    }

    pub fn clear(&mut self) {
        self.head = self.buffer.as_mut_ptr();
        self.size = 0;
    }
}

// Iterator implementation omitted for brevity

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {}
}
