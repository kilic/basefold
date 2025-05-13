pub mod basefold;
pub mod data;
pub mod field;
pub mod hash;
pub mod merkle;
pub mod mle;
pub mod utils;

#[derive(Debug, Clone, PartialEq, Eq, Copy)]
pub enum Error {
    Transcript,
    Verify,
}

#[cfg(test)]
pub(crate) mod test {
    use rand::{rngs::SmallRng, SeedableRng};

    #[allow(dead_code)]
    pub(crate) fn seed_rng() -> SmallRng {
        SmallRng::seed_from_u64(1)
    }

    #[allow(dead_code)]
    pub(crate) fn init_tracing() {
        use tracing_forest::util::LevelFilter;
        use tracing_forest::ForestLayer;
        use tracing_subscriber::layer::SubscriberExt;
        use tracing_subscriber::util::SubscriberInitExt;
        use tracing_subscriber::{EnvFilter, Registry};

        let env_filter = EnvFilter::builder()
            .with_default_directive(LevelFilter::INFO.into())
            .from_env_lossy();

        Registry::default()
            .with(env_filter)
            .with(ForestLayer::default())
            .init();
    }
}
