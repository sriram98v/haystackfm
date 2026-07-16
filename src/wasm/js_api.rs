use wasm_bindgen::prelude::*;

use crate::alphabet::DnaSequence;
use crate::fm_index::{FmIndex, FmIndexConfig};

/// Convert any error type to JsValue for wasm-bindgen.
fn to_js_err(e: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&e.to_string())
}

/// Parse a multi-line FASTA or plain-sequence string into a list of DNA sequences.
///
/// - Lines starting with `>` are treated as FASTA headers (stored in `DnaSequence`).
/// - Blank lines are skipped.
/// - Everything else is treated as sequence data (may span multiple lines per entry).
///
/// Returns one `DnaSequence` per FASTA record (or one per non-blank line for plain input).
fn parse_fasta(input: &str) -> Result<Vec<DnaSequence>, JsValue> {
    let mut sequences: Vec<DnaSequence> = Vec::new();
    let mut current_seq: Option<String> = None;
    let mut current_header: Option<String> = None;

    for line in input.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('>') {
            // Flush previous sequence.
            if let Some(seq_str) = current_seq.take() {
                if !seq_str.is_empty() {
                    let header = current_header.take().unwrap_or_default();
                    sequences.push(
                        DnaSequence::from_str_with_header(&seq_str, &header).map_err(to_js_err)?,
                    );
                }
            }
            current_header = Some(rest.trim().to_string());
            current_seq = Some(String::new());
        } else {
            match current_seq.as_mut() {
                Some(buf) => buf.push_str(line),
                None => {
                    // Plain (non-FASTA) input: each non-blank line is its own sequence.
                    sequences.push(DnaSequence::from_str(line).map_err(to_js_err)?);
                }
            }
        }
    }

    // Flush last FASTA record.
    if let Some(seq_str) = current_seq.take() {
        if !seq_str.is_empty() {
            let header = current_header.take().unwrap_or_default();
            sequences
                .push(DnaSequence::from_str_with_header(&seq_str, &header).map_err(to_js_err)?);
        }
    }

    if sequences.is_empty() {
        return Err(JsValue::from_str("no sequences found in input"));
    }

    Ok(sequences)
}

/// A builder that accumulates DNA sequences and constructs an FM-index.
///
/// ```js
/// const builder = new FmIndexBuilder(32);
/// builder.add_sequence("ACGTACGT");
/// const handle = await builder.build_gpu();  // GPU path
/// // or: const handle = builder.build_cpu(); // CPU path (sync)
/// ```
#[wasm_bindgen]
pub struct FmIndexBuilder {
    sequences: Vec<DnaSequence>,
    sa_sample_rate: u32,
}

#[wasm_bindgen]
impl FmIndexBuilder {
    /// Create a new builder.
    ///
    /// `sa_sample_rate`: controls the trade-off between locate speed and memory.
    /// Higher = less memory, slower locate queries. Default: 32.
    #[wasm_bindgen(constructor)]
    pub fn new(sa_sample_rate: Option<u32>) -> Self {
        Self {
            sequences: Vec::new(),
            sa_sample_rate: sa_sample_rate.unwrap_or(32),
        }
    }

    /// Add a single DNA sequence (ACGT characters, case-insensitive).
    pub fn add_sequence(&mut self, seq: &str) -> Result<(), JsValue> {
        let dna = DnaSequence::from_str(seq).map_err(to_js_err)?;
        self.sequences.push(dna);
        Ok(())
    }

    /// Add sequences from a FASTA string or newline-separated plain sequences.
    pub fn add_fasta(&mut self, fasta: &str) -> Result<(), JsValue> {
        let mut seqs = parse_fasta(fasta)?;
        self.sequences.append(&mut seqs);
        Ok(())
    }

    /// Number of sequences currently staged.
    pub fn sequence_count(&self) -> usize {
        self.sequences.len()
    }

    /// Clear all staged sequences.
    pub fn clear(&mut self) {
        self.sequences.clear();
    }

    /// Build the FM-index using the CPU (synchronous).
    pub fn build_cpu(&self) -> Result<FmIndexHandle, JsValue> {
        if self.sequences.is_empty() {
            return Err(JsValue::from_str("no sequences added"));
        }
        let config = FmIndexConfig {
            sa_sample_rate: self.sa_sample_rate,
            use_gpu: false,
            ..Default::default()
        };
        let index = FmIndex::build_cpu(&self.sequences, &config).map_err(to_js_err)?;
        Ok(FmIndexHandle { index })
    }

    /// Build the FM-index using the GPU (asynchronous, returns a Promise).
    ///
    /// Falls back gracefully: if GPU initialization fails the promise rejects
    /// with the error message.
    pub async fn build_gpu(&self) -> Result<FmIndexHandle, JsValue> {
        if self.sequences.is_empty() {
            return Err(JsValue::from_str("no sequences added"));
        }
        let config = FmIndexConfig {
            sa_sample_rate: self.sa_sample_rate,
            use_gpu: true,
            ..Default::default()
        };
        let index = FmIndex::build(&self.sequences, &config)
            .await
            .map_err(to_js_err)?;
        Ok(FmIndexHandle { index })
    }
}

/// A built FM-index, ready for queries.
///
/// Obtain one from `FmIndexBuilder.build_cpu()` or `await FmIndexBuilder.build_gpu()`.
#[wasm_bindgen]
pub struct FmIndexHandle {
    index: FmIndex,
}

#[wasm_bindgen]
impl FmIndexHandle {
    /// Count occurrences of `pattern` (ACGT string) in the indexed sequences.
    ///
    /// Returns 0 if the pattern is not found.
    pub fn count(&self, pattern: &str) -> Result<u32, JsValue> {
        let encoded = encode_pattern(pattern)?;
        Ok(self.index.count(&encoded))
    }

    /// Locate all occurrences of `pattern`.
    ///
    /// Returns a JS `Array` of `[sequenceId, position]` pairs, where `sequenceId`
    /// is the FASTA header string and `position` is 0-based within that sequence.
    pub fn locate(&self, pattern: &str) -> Result<js_sys::Array, JsValue> {
        let encoded = encode_pattern(pattern)?;
        let hits = self.index.locate(&encoded);
        let result = js_sys::Array::new_with_length(hits.len() as u32);
        for (i, (seq_id, pos)) in hits.into_iter().enumerate() {
            let pair = js_sys::Array::new_with_length(2);
            pair.set(0, wasm_bindgen::JsValue::from_str(&seq_id));
            pair.set(1, wasm_bindgen::JsValue::from_f64(pos as f64));
            result.set(i as u32, pair.into());
        }
        Ok(result)
    }

    /// Total length of the indexed text (including per-sequence sentinel characters).
    pub fn text_len(&self) -> u32 {
        self.index.text_len()
    }

    /// Number of sequences in the index.
    pub fn num_sequences(&self) -> u32 {
        self.index.num_sequences()
    }

    /// Serialize the FM-index to a `Uint8Array` for storage or transfer.
    pub fn to_bytes(&self) -> Result<Vec<u8>, JsValue> {
        self.index.to_bytes().map_err(to_js_err)
    }

    /// Deserialize an FM-index from bytes previously produced by `to_bytes()`.
    pub fn from_bytes(data: &[u8]) -> Result<FmIndexHandle, JsValue> {
        let index = FmIndex::from_bytes(data).map_err(to_js_err)?;
        Ok(FmIndexHandle { index })
    }
}

/// Encode an ACGT pattern string to internal byte representation.
fn encode_pattern(pattern: &str) -> Result<Vec<u8>, JsValue> {
    use crate::alphabet::encode_char;
    pattern
        .chars()
        .enumerate()
        .map(|(i, ch)| {
            encode_char(ch).ok_or_else(|| {
                JsValue::from_str(&format!("invalid DNA character '{}' at position {}", ch, i))
            })
        })
        .collect()
}
