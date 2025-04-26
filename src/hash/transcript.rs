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

pub trait GrindWriter<T>: Writer<T> + ChallengeBits + Clone {
    fn grind(&mut self, bits: usize) -> Result<(), crate::Error>
    where
        T: std::fmt::Debug + Clone + Copy + From<u64>,
    {
        let witness = (0u64..)
            .map(|i| i.into())
            .find(|&cand| self.check_witness_silent(bits, cand).unwrap())
            .unwrap();

        self.check_witness(bits, witness)?;

        Ok(())
    }

    fn check_witness(&mut self, bits: usize, witness: T) -> Result<bool, crate::Error>
    where
        T: std::fmt::Debug + Clone,
    {
        self.write(witness.clone())?;
        Ok(ChallengeBits::draw(self, bits) == 0)
    }

    fn check_witness_silent(&self, bits: usize, witness: T) -> Result<bool, crate::Error>
    where
        T: std::fmt::Debug + Clone,
    {
        let mut this = self.clone();
        this.check_witness(bits, witness)
    }
}

pub trait GrindReader<T> {
    fn check_witness(&mut self, bits: usize) -> Result<bool, crate::Error>
    where
        T: std::fmt::Debug + Clone,
        Self: Reader<T> + ChallengeBits,
    {
        let _ = self.read()?;
        Ok(ChallengeBits::draw(self, bits) == 0)
    }
}
