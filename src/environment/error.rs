//! Error types for the `environment` module.
//!
//! ## Design
//!
//! The `environment` module uses its own error enum. All public environment
//! APIs return `Result<T, EnvironmentError>`.
//!
//! At scheduler / system boundaries the
//! [`From<EnvironmentError> for ECSError`] implementation (defined in
//! `src/engine/error/mod.rs` behind `#[cfg(feature = "environment")]`)
//! converts the error into `ECSError::Environment`, preserving full
//! diagnostic context.

/// Errors that can arise when reading or writing the simulation environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EnvironmentError {
    /// A key was requested that was never registered.
    KeyNotFound(String),

    /// The registered type for a key does not match the type requested by the
    /// caller.
    TypeMismatch {
        /// The key whose type mismatched.
        key: String,
        /// The name of the type that was registered.
        expected: &'static str,
        /// The name of the type that was requested.
        actual: &'static str,
    },
}

impl std::fmt::Display for EnvironmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnvironmentError::KeyNotFound(k) => write!(f, "key not found: {k}"),
            EnvironmentError::TypeMismatch {
                key,
                expected,
                actual,
            } => write!(
                f,
                "type mismatch for key '{key}': expected {expected}, got {actual}"
            ),
        }
    }
}

impl std::error::Error for EnvironmentError {}

/// Convenience alias for environment-domain results.
pub type EnvironmentResult<T> = Result<T, EnvironmentError>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::error::ECSError;

    #[test]
    fn display_key_not_found() {
        let e = EnvironmentError::KeyNotFound("interest_rate".into());
        assert!(e.to_string().contains("interest_rate"));
    }

    #[test]
    fn display_type_mismatch() {
        let e = EnvironmentError::TypeMismatch {
            key: "tax_rate".into(),
            expected: "f32",
            actual: "f64",
        };
        let s = e.to_string();
        assert!(s.contains("tax_rate"));
        assert!(s.contains("f32"));
        assert!(s.contains("f64"));
    }

    #[test]
    fn into_ecs_error_preserves_variant() {
        let e = EnvironmentError::KeyNotFound("x".into());
        let ecs: ECSError = e.into();
        assert!(matches!(
            ecs,
            ECSError::Environment(EnvironmentError::KeyNotFound(_))
        ));
    }

    #[test]
    fn into_ecs_error_preserves_type_mismatch() {
        let e = EnvironmentError::TypeMismatch {
            key: "rate".into(),
            expected: "f32",
            actual: "f64",
        };
        let ecs: ECSError = e.into();
        assert!(matches!(
            ecs,
            ECSError::Environment(EnvironmentError::TypeMismatch { .. })
        ));
        // Verify the Display output retains the original message.
        assert!(ecs.to_string().contains("rate"));
        assert!(ecs.to_string().contains("f32"));
    }
}
