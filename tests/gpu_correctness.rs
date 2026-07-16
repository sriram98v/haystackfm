#![cfg(feature = "gpu")]

mod common;

use haystackfm::alphabet::*;
use haystackfm::bwt::cpu::build_bwt;
use haystackfm::bwt::gpu::BwtPipelines;
use haystackfm::gpu::prefix_sum::PrefixSumPipelines;
use haystackfm::gpu::radix_sort::RadixSortPipelines;
use haystackfm::gpu::GpuContext;
use haystackfm::occ::cpu::{build_occ_table, naive_rank};
use haystackfm::occ::gpu::OccPipelines;
use haystackfm::occ::OccEncoding;
use haystackfm::suffix_array::cpu::build_suffix_array;
use haystackfm::suffix_array::gpu::SaPipelines;

fn get_gpu_context() -> Option<GpuContext> {
    pollster::block_on(GpuContext::new()).ok()
}

// ===================== Prefix Sum Tests =====================

#[test]
fn gpu_prefix_sum_small() {
    let Some(ctx) = get_gpu_context() else {
        eprintln!("Skipping: no GPU available");
        return;
    };
    let pipelines = PrefixSumPipelines::new(&ctx);

    let input = vec![1u32, 2, 3, 4, 5];
    let buf = ctx.create_buffer_init("test_data", &input);

    pipelines.exclusive_prefix_sum(&ctx, &buf, input.len() as u32);

    let result = pollster::block_on(ctx.download_buffer(&buf, input.len() as u32));
    assert_eq!(
        result,
        vec![0, 1, 3, 6, 10],
        "exclusive prefix sum of [1,2,3,4,5]"
    );
}

#[test]
fn gpu_prefix_sum_power_of_two() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = PrefixSumPipelines::new(&ctx);

    let input: Vec<u32> = (1..=8).collect();
    let buf = ctx.create_buffer_init("test_data", &input);
    pipelines.exclusive_prefix_sum(&ctx, &buf, 8);
    let result = pollster::block_on(ctx.download_buffer(&buf, 8));
    // exclusive prefix sum of [1,2,3,4,5,6,7,8] = [0,1,3,6,10,15,21,28]
    assert_eq!(result, vec![0, 1, 3, 6, 10, 15, 21, 28]);
}

#[test]
fn gpu_prefix_sum_single() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = PrefixSumPipelines::new(&ctx);

    let buf = ctx.create_buffer_init("test_data", &[42u32]);
    pipelines.exclusive_prefix_sum(&ctx, &buf, 1);
    let result = pollster::block_on(ctx.download_buffer(&buf, 1));
    assert_eq!(result, vec![0]);
}

#[test]
fn gpu_prefix_sum_multi_block() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = PrefixSumPipelines::new(&ctx);

    // 1000 elements (> 512, needs 2 blocks)
    let n = 1000u32;
    let input: Vec<u32> = (0..n).map(|_| 1).collect();
    let buf = ctx.create_buffer_init("test_data", &input);
    pipelines.exclusive_prefix_sum(&ctx, &buf, n);
    let result = pollster::block_on(ctx.download_buffer(&buf, n));

    let expected: Vec<u32> = (0..n).collect();
    assert_eq!(
        result, expected,
        "prefix sum of all-ones should be [0,1,2,...,n-1]"
    );
}

#[test]
fn gpu_prefix_sum_large() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = PrefixSumPipelines::new(&ctx);

    // 10000 elements (multiple levels of hierarchy)
    let n = 10000u32;
    let input: Vec<u32> = (0..n).map(|i| (i % 7) + 1).collect();
    let buf = ctx.create_buffer_init("test_data", &input);
    pipelines.exclusive_prefix_sum(&ctx, &buf, n);
    let result = pollster::block_on(ctx.download_buffer(&buf, n));

    // Compute expected on CPU
    let mut expected = vec![0u32; n as usize];
    let mut sum = 0u32;
    for i in 0..n as usize {
        expected[i] = sum;
        sum += input[i];
    }
    assert_eq!(result, expected);
}

// ===================== Radix Sort Tests =====================

#[test]
fn gpu_radix_sort_small() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let prefix_sum = PrefixSumPipelines::new(&ctx);
    let radix_sort = RadixSortPipelines::new(&ctx);

    let keys = vec![5u32, 3, 8, 1, 4, 2, 7, 6];
    let vals: Vec<u32> = (0..keys.len() as u32).collect();
    let n = keys.len() as u32;

    let keys_a = ctx.create_buffer_init("keys_a", &keys);
    let vals_a = ctx.create_buffer_init("vals_a", &vals);
    let keys_b = ctx.create_buffer_empty("keys_b", n);
    let vals_b = ctx.create_buffer_empty("vals_b", n);

    let result = radix_sort.sort(&ctx, &prefix_sum, &keys_a, &vals_a, &keys_b, &vals_b, n);

    let (sorted_keys_buf, sorted_vals_buf) = match result {
        haystackfm::gpu::radix_sort::SortResult::InA => (&keys_a, &vals_a),
        haystackfm::gpu::radix_sort::SortResult::InB => (&keys_b, &vals_b),
    };

    let sorted_keys = pollster::block_on(ctx.download_buffer(sorted_keys_buf, n));
    let sorted_vals = pollster::block_on(ctx.download_buffer(sorted_vals_buf, n));

    assert_eq!(sorted_keys, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    // vals should be the original indices in sorted key order.
    // keys[6]=7 (val=6) and keys[7]=6 (val=7): sorted key 6 comes before key 7,
    // so val=7 appears before val=6.
    assert_eq!(sorted_vals, vec![3, 5, 1, 4, 0, 7, 6, 2]);
}

#[test]
fn gpu_radix_sort_already_sorted() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let prefix_sum = PrefixSumPipelines::new(&ctx);
    let radix_sort = RadixSortPipelines::new(&ctx);

    let keys: Vec<u32> = (0..16).collect();
    let vals: Vec<u32> = (0..16).collect();
    let n = 16;

    let keys_a = ctx.create_buffer_init("keys_a", &keys);
    let vals_a = ctx.create_buffer_init("vals_a", &vals);
    let keys_b = ctx.create_buffer_empty("keys_b", n);
    let vals_b = ctx.create_buffer_empty("vals_b", n);

    let result = radix_sort.sort(&ctx, &prefix_sum, &keys_a, &vals_a, &keys_b, &vals_b, n);

    let sorted_keys_buf = match result {
        haystackfm::gpu::radix_sort::SortResult::InA => &keys_a,
        haystackfm::gpu::radix_sort::SortResult::InB => &keys_b,
    };
    let sorted_keys = pollster::block_on(ctx.download_buffer(sorted_keys_buf, n));
    assert_eq!(sorted_keys, keys);
}

#[test]
fn gpu_radix_sort_large_values() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let prefix_sum = PrefixSumPipelines::new(&ctx);
    let radix_sort = RadixSortPipelines::new(&ctx);

    // Keys spanning multiple bytes
    let keys = vec![1000u32, 500, 70000, 255, 256, 65536, 1, 0];
    let vals: Vec<u32> = (0..keys.len() as u32).collect();
    let n = keys.len() as u32;

    let keys_a = ctx.create_buffer_init("keys_a", &keys);
    let vals_a = ctx.create_buffer_init("vals_a", &vals);
    let keys_b = ctx.create_buffer_empty("keys_b", n);
    let vals_b = ctx.create_buffer_empty("vals_b", n);

    let result = radix_sort.sort(&ctx, &prefix_sum, &keys_a, &vals_a, &keys_b, &vals_b, n);

    let sorted_keys_buf = match result {
        haystackfm::gpu::radix_sort::SortResult::InA => &keys_a,
        haystackfm::gpu::radix_sort::SortResult::InB => &keys_b,
    };
    let sorted_keys = pollster::block_on(ctx.download_buffer(sorted_keys_buf, n));

    let mut expected = keys.clone();
    expected.sort();
    assert_eq!(sorted_keys, expected);
}

// ===================== Suffix Array Tests =====================

fn encode(s: &str) -> Vec<u8> {
    let mut v: Vec<u8> = s.chars().map(|c| encode_char(c).unwrap()).collect();
    if v.last() != Some(&SENTINEL) {
        v.push(SENTINEL);
    }
    v
}

#[test]
fn gpu_sa_matches_cpu_basic() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let text = encode("ACGT");
    let cpu_sa = build_suffix_array(&text);
    let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));

    assert_eq!(
        gpu_sa.data, cpu_sa.data,
        "GPU SA should match CPU SA for 'ACGT$'"
    );
}

#[test]
fn gpu_sa_matches_cpu_repeated() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let text = encode("AAAA");
    let cpu_sa = build_suffix_array(&text);
    let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));

    assert_eq!(
        gpu_sa.data, cpu_sa.data,
        "GPU SA should match CPU SA for 'AAAA$'"
    );
}

#[test]
fn gpu_sa_matches_cpu_longer() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let text = encode("ACGTACGTACGT");
    let cpu_sa = build_suffix_array(&text);
    let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));

    assert_eq!(
        gpu_sa.data, cpu_sa.data,
        "GPU SA should match CPU SA for 'ACGTACGTACGT$'"
    );
}

#[test]
fn gpu_sa_matches_cpu_various() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let inputs = [
        "A",
        "ACGT",
        "AAAA",
        "TGCATGCA",
        "ACGTTAGCCAGTACGT",
        "ACACACACAC",
        "GCGCGCGCGC",
    ];

    for input in &inputs {
        let text = encode(input);
        let cpu_sa = build_suffix_array(&text);
        let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));

        assert_eq!(
            gpu_sa.data, cpu_sa.data,
            "GPU SA should match CPU SA for '{}'",
            input
        );
    }
}

#[test]
fn gpu_sa_is_valid_permutation() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let text = encode("ACGTTAGCCAGTACGT");
    let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));
    let n = gpu_sa.len();

    let mut sorted = gpu_sa.data.clone();
    sorted.sort();
    assert_eq!(sorted, (0..n as u32).collect::<Vec<_>>());
}

#[test]
fn gpu_sa_produces_sorted_suffixes() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let pipelines = SaPipelines::new(&ctx);

    let text = encode("ACGTTAGCCAGTACGT");
    let gpu_sa = pollster::block_on(pipelines.build_suffix_array(&ctx, &text));

    for i in 1..gpu_sa.len() {
        let a = gpu_sa.data[i - 1] as usize;
        let b = gpu_sa.data[i] as usize;
        assert!(text[a..] < text[b..], "SA not sorted at position {}", i);
    }
}

// ===================== BWT Tests =====================

#[test]
fn gpu_bwt_matches_cpu_basic() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);

    let text = encode("ACGT");
    let cpu_sa = build_suffix_array(&text);
    let cpu_bwt = build_bwt(&text, &cpu_sa);

    let gpu_sa = pollster::block_on(sa_pipelines.build_suffix_array(&ctx, &text));
    let gpu_bwt = pollster::block_on(bwt_pipelines.build_bwt(&ctx, &text, &gpu_sa));

    assert_eq!(gpu_bwt, cpu_bwt, "GPU BWT should match CPU BWT for 'ACGT$'");
}

#[test]
fn gpu_bwt_matches_cpu_various() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);

    let inputs = ["ACGT", "AAAA", "TGCATGCA", "ACGTTAGCCAGTACGT", "ACACACACAC"];

    for input in &inputs {
        let text = encode(input);
        let cpu_sa = build_suffix_array(&text);
        let cpu_bwt = build_bwt(&text, &cpu_sa);

        let gpu_sa = pollster::block_on(sa_pipelines.build_suffix_array(&ctx, &text));
        let gpu_bwt = pollster::block_on(bwt_pipelines.build_bwt(&ctx, &text, &gpu_sa));

        assert_eq!(gpu_bwt, cpu_bwt, "GPU BWT mismatch for '{}'", input);
    }
}

// ===================== Occ Table Tests =====================

#[test]
fn gpu_occ_matches_cpu_basic() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);
    let occ_pipelines = OccPipelines::new(&ctx);

    let text = encode("ACGT");
    let gpu_sa = pollster::block_on(sa_pipelines.build_suffix_array(&ctx, &text));
    let gpu_bwt = pollster::block_on(bwt_pipelines.build_bwt(&ctx, &text, &gpu_sa));
    let gpu_occ = pollster::block_on(occ_pipelines.build_occ_table(&ctx, &gpu_bwt));

    let n = gpu_bwt.len() as u32;
    for c in 0..ALPHABET_SIZE as u8 {
        for i in 0..=n {
            let expected = naive_rank(&gpu_bwt, c, i);
            let actual = gpu_occ.rank(c, i);
            assert_eq!(
                actual, expected,
                "GPU Occ({}, {}) = {} but expected {} for 'ACGT$'",
                c, i, actual, expected
            );
        }
    }
}

#[test]
fn gpu_occ_matches_cpu_various() {
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);
    let occ_pipelines = OccPipelines::new(&ctx);

    let inputs = ["ACGT", "AAAA", "TGCATGCA", "ACGTTAGCCAGTACGT", "ACACACACAC"];

    for input in &inputs {
        let text = encode(input);
        let gpu_sa = pollster::block_on(sa_pipelines.build_suffix_array(&ctx, &text));
        let gpu_bwt = pollster::block_on(bwt_pipelines.build_bwt(&ctx, &text, &gpu_sa));
        let gpu_occ = pollster::block_on(occ_pipelines.build_occ_table(&ctx, &gpu_bwt));
        let cpu_occ = build_occ_table(&gpu_bwt, OccEncoding::Bitplane);

        let n = gpu_bwt.len() as u32;
        for c in 0..ALPHABET_SIZE as u8 {
            for i in 0..=n {
                assert_eq!(
                    gpu_occ.rank(c, i),
                    cpu_occ.rank(c, i),
                    "GPU/CPU Occ({}, {}) mismatch for '{}'",
                    c,
                    i,
                    input
                );
            }
        }
    }
}

#[test]
fn gpu_occ_multi_block() {
    // Text longer than one block (BLOCK_SIZE=64)
    let Some(ctx) = get_gpu_context() else {
        return;
    };
    let sa_pipelines = SaPipelines::new(&ctx);
    let bwt_pipelines = BwtPipelines::new(&ctx);
    let occ_pipelines = OccPipelines::new(&ctx);

    let s = "ACGT".repeat(20); // 80 chars + sentinel = 81 BWT chars -> 2 blocks
    let text = encode(&s);
    let gpu_sa = pollster::block_on(sa_pipelines.build_suffix_array(&ctx, &text));
    let gpu_bwt = pollster::block_on(bwt_pipelines.build_bwt(&ctx, &text, &gpu_sa));
    let gpu_occ = pollster::block_on(occ_pipelines.build_occ_table(&ctx, &gpu_bwt));
    let cpu_occ = build_occ_table(&gpu_bwt, OccEncoding::Bitplane);

    let n = gpu_bwt.len() as u32;
    for c in 0..ALPHABET_SIZE as u8 {
        for i in 0..=n {
            assert_eq!(
                gpu_occ.rank(c, i),
                cpu_occ.rank(c, i),
                "GPU/CPU Occ({}, {}) mismatch for multi-block text",
                c,
                i
            );
        }
    }
}

// ===================== Full GPU Pipeline Tests =====================

#[test]
fn gpu_fm_index_count_matches_cpu() {
    use haystackfm::{DnaSequence, FmIndex, FmIndexConfig};

    let Some(_) = get_gpu_context() else {
        return;
    };

    let seqs = vec![DnaSequence::from_str("ACGTACGTACGT").unwrap()];
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: true,
        ..Default::default()
    };

    let cpu = FmIndex::build_cpu(&seqs, &config).unwrap();
    let gpu = pollster::block_on(FmIndex::build(&seqs, &config)).unwrap();

    // count queries must match
    for pattern in &["ACG", "CGT", "AAA", "ACGT", "T"] {
        let p: Vec<u8> = pattern.chars().map(|c| encode_char(c).unwrap()).collect();
        assert_eq!(
            cpu.count(&p),
            gpu.count(&p),
            "count mismatch for pattern {:?}",
            pattern
        );
    }
}

#[test]
fn gpu_fm_index_locate_matches_cpu() {
    use haystackfm::{DnaSequence, FmIndex, FmIndexConfig};

    let Some(_) = get_gpu_context() else {
        return;
    };

    let seqs = vec![DnaSequence::from_str("ACGTACGTACGT").unwrap()];
    let config = FmIndexConfig {
        sa_sample_rate: 1,
        use_gpu: true,
        ..Default::default()
    };

    let cpu = FmIndex::build_cpu(&seqs, &config).unwrap();
    let gpu = pollster::block_on(FmIndex::build(&seqs, &config)).unwrap();

    for pattern in &["ACG", "CGT", "ACGT"] {
        let p: Vec<u8> = pattern.chars().map(|c| encode_char(c).unwrap()).collect();
        let mut cpu_locs = cpu.locate(&p);
        let mut gpu_locs = gpu.locate(&p);
        cpu_locs.sort();
        gpu_locs.sort();
        assert_eq!(
            cpu_locs, gpu_locs,
            "locate mismatch for pattern {:?}",
            pattern
        );
    }
}
