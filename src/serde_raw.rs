//! `#[serde(with = ...)]` codecs that store a numeric vector as one length-prefixed
//! little-endian byte blob instead of an element sequence.
//!
//! Under `bincode` 1 the derived encoding of a `Vec<u32>` is `u64 count` followed by the
//! elements, and serde's `Vec` visitor reads them back one `Deserializer` call at a time
//! (with a capped preallocation and doubling growth). A byte blob is `u64 byte-len` followed
//! by the raw bytes, which `bincode`'s slice reader hands to the visitor as a single slice:
//! one length check and one copy, however large the vector.
//!
//! On the wire, `bytes` is identical to the derived encoding of a `Vec<u8>`; `u16s` and
//! `u32s` differ from the derived encoding only in the length prefix (bytes, not elements),
//! which is why using them is a format-version change (see `fm_index::serialize`).
//! Elements are always written little-endian, so blobs are portable across hosts.

use serde::de::{self, Visitor};
use serde::{Deserializer, Serializer};
use std::fmt;

/// `Vec<u8>` / `&[u8]` as a byte blob.
pub(crate) mod bytes {
    use super::*;

    pub(crate) fn serialize<T, S>(v: &T, s: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u8]> + ?Sized,
        S: Serializer,
    {
        s.serialize_bytes(v.as_ref())
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
        d.deserialize_bytes(BlobVisitor {
            width: 1,
            decode: |b| b.to_vec(),
        })
    }
}

/// `Vec<u16>` as a little-endian byte blob.
pub(crate) mod u16s {
    use super::*;

    pub(crate) fn serialize<T, S>(v: &T, s: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u16]> + ?Sized,
        S: Serializer,
    {
        let v = v.as_ref();
        let mut out = Vec::with_capacity(v.len() * 2);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        s.serialize_bytes(&out)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u16>, D::Error> {
        d.deserialize_bytes(BlobVisitor {
            width: 2,
            decode: |b| {
                b.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect()
            },
        })
    }
}

/// `Vec<u32>` as a little-endian byte blob.
pub(crate) mod u32s {
    use super::*;

    pub(crate) fn serialize<T, S>(v: &T, s: S) -> Result<S::Ok, S::Error>
    where
        T: AsRef<[u32]> + ?Sized,
        S: Serializer,
    {
        let v = v.as_ref();
        let mut out = Vec::with_capacity(v.len() * 4);
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
        s.serialize_bytes(&out)
    }

    pub(crate) fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u32>, D::Error> {
        d.deserialize_bytes(BlobVisitor {
            width: 4,
            decode: |b| {
                b.chunks_exact(4)
                    .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect()
            },
        })
    }
}

/// Accepts a byte blob whose length is a multiple of `width` and decodes it in one pass.
/// `visit_borrowed_bytes` / `visit_byte_buf` fall through to `visit_bytes` by default, so
/// this works for borrowing and owning deserializers alike.
struct BlobVisitor<T, F: Fn(&[u8]) -> Vec<T>> {
    width: usize,
    decode: F,
}

impl<'de, T, F: Fn(&[u8]) -> Vec<T>> Visitor<'de> for BlobVisitor<T, F> {
    type Value = Vec<T>;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "a little-endian byte blob whose length is a multiple of {}",
            self.width
        )
    }

    fn visit_bytes<E: de::Error>(self, b: &[u8]) -> Result<Vec<T>, E> {
        if !b.len().is_multiple_of(self.width) {
            return Err(E::invalid_length(b.len(), &self));
        }
        Ok((self.decode)(b))
    }
}

#[cfg(test)]
mod tests {
    use serde::{Deserialize, Serialize};

    #[derive(Serialize, Deserialize, PartialEq, Debug)]
    struct Raw {
        #[serde(with = "crate::serde_raw::bytes")]
        a: Vec<u8>,
        #[serde(with = "crate::serde_raw::u16s")]
        b: Vec<u16>,
        #[serde(with = "crate::serde_raw::u32s")]
        c: Vec<u32>,
    }

    #[derive(Serialize)]
    struct RawBorrowed<'a> {
        #[serde(with = "crate::serde_raw::bytes")]
        a: &'a [u8],
        #[serde(with = "crate::serde_raw::u16s")]
        b: &'a [u16],
        #[serde(with = "crate::serde_raw::u32s")]
        c: &'a [u32],
    }

    #[derive(Serialize)]
    struct Derived {
        a: Vec<u8>,
        b: Vec<u16>,
        c: Vec<u32>,
    }

    fn sample(n: usize) -> Raw {
        Raw {
            a: (0..n).map(|i| (i * 7) as u8).collect(),
            b: (0..n).map(|i| (i * 300) as u16).collect(),
            c: (0..n)
                .map(|i| (i as u32).wrapping_mul(0x9E37_79B9))
                .collect(),
        }
    }

    #[test]
    fn round_trips_every_width_and_length() {
        for n in [0, 1, 3, 64, 1000] {
            let raw = sample(n);
            let bytes = bincode::serialize(&raw).unwrap();
            let back: Raw = bincode::deserialize(&bytes).unwrap();
            assert_eq!(back, raw, "n = {n}");
            let borrowed = RawBorrowed {
                a: &raw.a,
                b: &raw.b,
                c: &raw.c,
            };
            assert_eq!(bincode::serialize(&borrowed).unwrap(), bytes, "n = {n}");
        }
    }

    #[test]
    fn byte_blob_is_wire_identical_to_the_derived_vec_u8() {
        let raw = sample(100);
        let derived = Derived {
            a: raw.a.clone(),
            b: Vec::new(),
            c: Vec::new(),
        };
        let only_a = Raw {
            a: raw.a.clone(),
            b: Vec::new(),
            c: Vec::new(),
        };
        assert_eq!(
            bincode::serialize(&only_a).unwrap(),
            bincode::serialize(&derived).unwrap()
        );
    }

    #[test]
    fn wide_blobs_are_byte_length_prefixed_little_endian() {
        let raw = Raw {
            a: Vec::new(),
            b: vec![0x0102],
            c: vec![0x0304_0506],
        };
        let bytes = bincode::serialize(&raw).unwrap();
        let expected: Vec<u8> = [
            0u64.to_le_bytes().as_slice(),
            2u64.to_le_bytes().as_slice(),
            &[0x02, 0x01],
            4u64.to_le_bytes().as_slice(),
            &[0x06, 0x05, 0x04, 0x03],
        ]
        .concat();
        assert_eq!(bytes, expected);
    }

    #[test]
    fn rejects_blob_length_not_a_multiple_of_the_width() {
        // `c` claims 5 bytes.
        let bytes: Vec<u8> = [
            0u64.to_le_bytes().as_slice(),
            0u64.to_le_bytes().as_slice(),
            5u64.to_le_bytes().as_slice(),
            &[1, 2, 3, 4, 5],
        ]
        .concat();
        let err = bincode::deserialize::<Raw>(&bytes).unwrap_err();
        assert!(err.to_string().contains("multiple of 4"), "{err}");
    }
}
