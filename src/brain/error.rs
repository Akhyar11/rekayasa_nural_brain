use std::fmt;
use std::io;
use std::error::Error;

#[derive(Debug)]
pub enum BrainError {
    EmptyInput,
    InvalidConfig(&'static str),
    Io(io::Error),
    Encode(bincode::error::EncodeError),
    Decode(bincode::error::DecodeError),
    SerdeJson(serde_json::Error),
    MissingToken(u64),
    MissingNode(u64),
    UnsupportedStateVersion(u32),
}

impl fmt::Display for BrainError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "input tidak boleh kosong"),
            Self::InvalidConfig(message) => write!(f, "{message}"),
            Self::Io(error) => write!(f, "{error}"),
            Self::Encode(error) => write!(f, "{error}"),
            Self::Decode(error) => write!(f, "{error}"),
            Self::SerdeJson(error) => write!(f, "{error}"),
            Self::MissingToken(node_id) => {
                write!(f, "token untuk node #{node_id} tidak ditemukan")
            }
            Self::MissingNode(node_id) => write!(f, "node #{node_id} tidak ditemukan"),
            Self::UnsupportedStateVersion(version) => {
                write!(f, "versi state {version} tidak didukung")
            }
        }
    }
}

impl Error for BrainError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Encode(error) => Some(error),
            Self::Decode(error) => Some(error),
            Self::SerdeJson(error) => Some(error),
            _ => None,
        }
    }
}

impl From<io::Error> for BrainError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<bincode::error::EncodeError> for BrainError {
    fn from(error: bincode::error::EncodeError) -> Self {
        Self::Encode(error)
    }
}

impl From<bincode::error::DecodeError> for BrainError {
    fn from(error: bincode::error::DecodeError) -> Self {
        Self::Decode(error)
    }
}

impl From<serde_json::Error> for BrainError {
    fn from(error: serde_json::Error) -> Self {
        Self::SerdeJson(error)
    }
}
