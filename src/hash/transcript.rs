pub trait Challenge<F> {
    fn draw(&mut self) -> F;
    fn draw_n(&mut self, n: usize) -> Vec<F> {
        (0..n).map(|_| self.draw()).collect()
    }
}

pub trait ChallengeBits {
    fn draw(&mut self, bit_size: usize) -> usize;
    fn draw_n(&mut self, n: usize, bit_size: usize) -> Vec<usize> {
        (0..n).map(|_| self.draw(bit_size)).collect()
    }
}

pub trait Writer<T> {
    fn write(&mut self, el: T) -> Result<(), crate::Error>;
    fn unsafe_write(&mut self, el: T) -> Result<(), crate::Error>;
    fn write_many(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.write(e))
    }
    fn unsafe_write_many(&mut self, el: &[T]) -> Result<(), crate::Error>
    where
        T: Copy,
    {
        el.iter().try_for_each(|&e| self.unsafe_write(e))
    }
}

pub trait Reader<T> {
    fn read(&mut self) -> Result<T, crate::Error>;
    fn unsafe_read(&mut self) -> Result<T, crate::Error>;
    fn read_many(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n).map(|_| self.read()).collect::<Result<Vec<_>, _>>()
    }
    fn unsafe_read_many(&mut self, n: usize) -> Result<Vec<T>, crate::Error> {
        (0..n)
            .map(|_| self.unsafe_read())
            .collect::<Result<Vec<_>, _>>()
    }
}
