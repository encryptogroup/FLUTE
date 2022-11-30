//! Puncturable pseudorandom function.
//!
//! This implementation and comments closely follows the one in libOTe.
use crate::util::aes_hash::FIXED_KEY_HASH;
use crate::util::aes_rng::AesRng;
use crate::util::tokio_rayon::{spawn_compute, AsyncThreadPool};
use crate::util::transpose::transpose;
use crate::util::{log2_ceil, Block};
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use bitvec::vec::BitVec;
use futures::FutureExt;
use ndarray::Array2;
use rand::Rng;
use rand_core::{CryptoRng, RngCore, SeedableRng};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use rayon::ThreadPool;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Debug;
use std::num::NonZeroUsize;
use std::sync::{Arc, Mutex};
use std::thread::available_parallelism;
use std::{cmp, mem};
use thiserror::Error;

pub struct Sender {
    conf: PprfConfig,
    base_ots: Array2<[Block; 2]>,
}

pub struct Receiver {
    conf: PprfConfig,
    base_ots: Array2<Block>,
    base_choices: Array2<u8>,
}

#[derive(Copy, Clone)]
pub struct PprfConfig {
    pnt_count: usize,
    domain: usize,
    depth: usize,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum Msg {
    TreeGrp(TreeGrp),
}

#[derive(Copy, Clone, Debug)]
pub enum PprfOutputFormat {
    Plain,
    InterleavedTransposed,
}

#[derive(Debug)]
pub struct ChoiceBits(Array2<u8>);

impl Sender {
    pub fn new(conf: PprfConfig, base_ots: Vec<[Block; 2]>) -> Self {
        assert_eq!(conf.base_ot_count(), base_ots.len());
        let base_ots = Array2::from_shape_vec((conf.pnt_count, conf.depth), base_ots).unwrap();
        Self { conf, base_ots }
    }

    pub async fn expand<RNG>(
        &mut self,
        sender: mpc_channel::Sender<Msg>,
        value: Block,
        rng: &mut RNG,
        thread_pool: Option<Arc<ThreadPool>>,
    ) -> Array2<Block>
    where
        RNG: RngCore + CryptoRng,
    {
        let conf = self.conf;
        let num_threads = thread_pool
            .as_ref()
            .map(|pool| pool.current_num_threads())
            .unwrap_or_else(|| {
                available_parallelism()
                    .unwrap_or(NonZeroUsize::new(1).unwrap())
                    .into()
            });
        let cols = (conf.pnt_count * conf.domain + 127) / 128;
        let output = Arc::new(Mutex::new(Array2::zeros((128, cols))));
        let aes = create_fixed_aes();
        let seed: Block = rng.gen();

        // let (sender, mut receiver) = mpsc::unbounded_channel();
        // let send_task = async move {
        //     while let Some(tree_grp) = receiver.recv().await {
        //         channel
        //             .send(Msg::TreeGrp(tree_grp))
        //             .await
        //             .map_err(|_| {})
        //             .unwrap();
        //     }
        //     Result::<(), ()>::Ok(())
        // };
        let base_ots = mem::take(&mut self.base_ots);
        let depth = conf.depth;
        let pnt_count = conf.pnt_count;
        let output_clone = Arc::clone(&output);
        let sender_cl = sender.clone();
        let routine = move |thread_idx: usize| {
            let mut rng = AesRng::from_seed(seed ^ thread_idx.into());
            let dd = depth + 1;
            // tree will hold the full GGM tree. Note that there are 8
            // independent trees that are being processed together.
            // The trees are flattened to that the children of j are
            // located at 2*j  and 2*j+1.
            let mut tree = vec![[Block::zero(); 8]; 2_usize.pow(dd as u32)];

            for g in (thread_idx * 8..pnt_count).step_by(8 * num_threads) {
                let mut tree_grp = TreeGrp {
                    g,
                    ..Default::default()
                };
                // The number of real trees for this iteration.
                let min = 8.min(pnt_count - g);
                let level = bytemuck::cast_slice_mut(get_level(&mut tree, 0));
                // Populate the zero'th level of the GGM tree with random seeds.
                rng.fill_bytes(level);
                tree_grp.sums[0].resize(depth, Default::default());
                tree_grp.sums[1].resize(depth, Default::default());

                for d in 0..depth {
                    let (level0, level1) = get_cons_levels(&mut tree, d);
                    let width = level1.len();
                    let mut child_idx = 0;
                    // For each child, populate the child by expanding the parent.
                    while child_idx < width {
                        // Index of the parent in the previous level.
                        let parent_idx = child_idx >> 1;
                        // The value of the parent.
                        let parent = &mut level0[parent_idx];
                        // The bit that indicates if we are on the left child (0)
                        // or on the right child (1).
                        let mut keep = 0;
                        while keep < 2 {
                            // The child that we will write in this iteration.
                            let child = &mut level1[child_idx];
                            // The sum that this child node belongs to.
                            let sum = &mut tree_grp.sums[keep][d];
                            // Each parent is expanded into the left and right children
                            // using a different AES fixed-key. Therefore our OWF is:
                            //
                            //    H(x) = (AES(k0, x) + x) || (AES(k1, x) + x);
                            //
                            // where each half defines one of the children.
                            aes[keep]
                                .encrypt_blocks_b2b(
                                    Block::cast_slice(&*parent),
                                    Block::cast_slice_mut(child),
                                )
                                .expect("Unequal block length is impossible");

                            child.iter_mut().zip(&mut *parent).for_each(|(c, p)| {
                                *c ^= *p;
                            });
                            // Update the running sums for this level. We keep
                            // a left and right totals for each level.
                            sum.iter_mut().zip(child).for_each(|(s, c)| {
                                *s ^= *c;
                            });
                            keep += 1;
                            child_idx += 1;
                        }
                    }
                }

                // For all but the last level, mask the sums with the
                // OT strings and send them over.

                let mut mask_sums = |idx: usize| {
                    tree_grp.sums[idx]
                        .iter_mut()
                        .take(depth - 1)
                        .enumerate()
                        .for_each(|(d, sums)| {
                            sums.iter_mut()
                                .enumerate()
                                .take(min as usize)
                                .for_each(|(j, sum)| *sum ^= base_ots[[g + j, d]][idx])
                        });
                };
                mask_sums(0);
                mask_sums(1);

                // For the last level, we are going to do something special.
                // The other party is currently missing both leaf children of
                // the active parent. Since this is the last level, we want
                // the inactive child to just be the normal value but the
                // active child should be the correct value XOR the delta.
                // This will be done by sending the sums and the sums plus
                // delta and ensure that they can only decrypt the correct ones.

                let d = depth - 1;
                tree_grp.last_ots.resize(min as usize, Default::default());
                for j in 0..min as usize {
                    // Construct the sums where we will allow the delta (mValue)
                    // to either be on the left child or right child depending
                    // on which has the active path.
                    tree_grp.last_ots[j][0] = tree_grp.sums[0][d][j];
                    tree_grp.last_ots[j][1] = tree_grp.sums[1][d][j] ^ value;
                    tree_grp.last_ots[j][2] = tree_grp.sums[1][d][j];
                    tree_grp.last_ots[j][3] = tree_grp.sums[0][d][j] ^ value;

                    // We are going to expand the two 128 bit OT strings
                    // into 256 bit OT strings using AES.
                    let mask_in = [
                        base_ots[[g + j, d]][0],
                        base_ots[[g + j, d]][0] ^ Block::all_ones(),
                        base_ots[[g + j, d]][1],
                        base_ots[[g + j, d]][1] ^ Block::all_ones(),
                    ];
                    let masks = FIXED_KEY_HASH.cr_hash_blocks(&mask_in);

                    // Add the OT masks to the sums and send them over.
                    tree_grp.last_ots[j]
                        .iter_mut()
                        .zip(masks)
                        .for_each(|(ot, mask)| {
                            *ot ^= mask;
                        });
                }
                // Resize the sums to that they dont include
                // the unmasked sums on the last level!
                tree_grp.sums[0].truncate(depth - 1);
                tree_grp.sums[1].truncate(depth - 1);

                sender_cl
                    .blocking_send(Msg::TreeGrp(tree_grp.clone()))
                    .expect("Sending tree group failed");
                let last_level = get_level(&mut tree, depth);
                let mut output = output_clone.lock().unwrap();
                copy_out(last_level, &mut output, pnt_count, g);
            }
        };

        let par_compute = move || {
            // TODO: this changes the meaning of num_threads. By using par_iter, it becomes the
            //  maximum number of threads
            (0..num_threads).into_par_iter().for_each(routine);
        };
        match thread_pool {
            None => spawn_compute(par_compute),
            Some(pool) => pool.spawn_install_compute(par_compute),
        }
        .await;

        let output = Arc::try_unwrap(output).unwrap();
        output.into_inner().unwrap()
    }
}

impl Receiver {
    pub fn new(conf: PprfConfig, base_ots: Vec<Block>, base_choices: ChoiceBits) -> Self {
        assert_eq!(conf.base_ot_count(), base_ots.len());
        assert_eq!(conf.base_ot_count(), base_choices.0.len());
        let base_ots = Array2::from_shape_vec((conf.pnt_count, conf.depth), base_ots).unwrap();
        Self {
            conf,
            base_ots,
            base_choices: base_choices.0,
        }
    }

    pub async fn expand(
        &mut self,
        mut receiver: mpc_channel::Receiver<Msg>,
        thread_pool: Option<Arc<ThreadPool>>,
    ) -> Array2<Block> {
        let conf = self.conf;
        let num_threads = thread_pool
            .as_ref()
            .map(|pool| pool.current_num_threads())
            .unwrap_or_else(|| {
                available_parallelism()
                    .unwrap_or(NonZeroUsize::new(1).unwrap())
                    .into()
            });

        let cols = (conf.pnt_count * conf.domain + 127) / 128;
        let output = Arc::new(Mutex::new(Array2::zeros((128, cols))));

        let points = self.get_points(PprfOutputFormat::Plain);
        let aes = create_fixed_aes();

        let base_ots = mem::take(&mut self.base_ots);
        let base_choices = mem::take(&mut self.base_choices);
        let depth = conf.depth;
        let pnt_count = conf.pnt_count;
        let output_clone = Arc::clone(&output);
        let expected_trees = pnt_count / 8;

        let (dist_sender, distributor) = crossbeam_channel::unbounded();
        // this task distributes the tree groups to the compute tasks
        let distribute_task = Box::pin(
            async {
                let mut received_items = 0;
                while let Some(msg) = receiver.recv().await.transpose() {
                    match msg {
                        Ok(Msg::TreeGrp(tree_grp)) => {
                            received_items += 1;
                            dist_sender.try_send(tree_grp).unwrap();
                        }
                        _ => panic!("Error receiving msg"),
                    };
                    // Only take as many tree_grp as are expected
                    if received_items == expected_trees {
                        break;
                    }
                }
            }
            .fuse(),
        );

        let routine = move |thread_idx: usize| {
            // my_sums will hold the left and right GGM tree sums
            // for each level. For example my_sums[5][0]  will
            // hold the sum of the left children for the 5th tree. This
            // sum will be "missing" the children of the active parent.
            // The sender will give of one of the full sums so we can
            // compute the missing inactive child.
            let mut my_sums = [[Block::zero(); 8]; 2];

            // // A buffer for receiving the sums from the other party.
            // // These will be masked by the OT strings.
            // let mut their_sums: [Vec<[Block; 8]>; 2] = Default::default();
            // their_sums[0].resize(depth - 1, [Block::zero(); 8]);
            // their_sums[1].resize(depth - 1, [Block::zero(); 8]);

            let dd = depth + 1;

            let mut tree = vec![[Block::zero(); 8]; 1 << dd];

            (thread_idx * 8..pnt_count)
                .step_by(8 * num_threads)
                .for_each(|_| {
                    let tree_group = distributor.recv().unwrap();
                    let g = tree_group.g;

                    let l1 = get_level(&mut tree, 1);
                    for i in 0..8 {
                        // For the non-active path, set the child of the root node
                        // as the OT message XOR'ed with the correction sum.
                        let not_ai = base_choices[[i + g, 0]] as usize;
                        l1[not_ai][i] = base_ots[[i + g, 0]] ^ tree_group.sums[not_ai][0][i];
                        // not_ai is either 0 or 1, so we flip it
                        l1[not_ai ^ 1][i] = Block::zero();
                    }

                    // For all other levels, expand the GGM tree and add in
                    // the correction along the active path.
                    for d in 1..depth {
                        // level0: The already constructed level. Only missing the
                        //          GGM tree node value along the active path.
                        // level1: The next level that we want to construct.
                        let (level0, level1) = get_cons_levels(&mut tree, d);
                        // Zero out the previous sums.
                        my_sums = [[Block::zero(); 8]; 2];
                        // We will iterate over each node on this level and
                        // expand it into it's two children. Note that the
                        // active node will also be expanded. Later we will just
                        // overwrite whatever the value was. This is an optimization.
                        let width = level1.len();
                        let mut child_idx = 0;
                        while child_idx < width {
                            // Index of the parent in the previous level.
                            let parent_idx = child_idx >> 1;
                            // The value of the parent.
                            let parent = &mut level0[parent_idx];
                            // The bit that indicates if we are on the left child (0)
                            // or on the right child (1).
                            let mut keep = 0;
                            while keep < 2 {
                                // The child that we will write in this iteration.
                                let child = &mut level1[child_idx];
                                // Each parent is expanded into the left and right children
                                // using a different AES fixed-key. Therefore our OWF is:
                                //
                                //    H(x) = (AES(k0, x) + x) || (AES(k1, x) + x);
                                //
                                // where each half defines one of the children.
                                aes[keep]
                                    .encrypt_blocks_b2b(
                                        Block::cast_slice(&*parent),
                                        Block::cast_slice_mut(child),
                                    )
                                    .expect("Unequal block length is impossible");

                                child.iter_mut().zip(&mut *parent).for_each(|(c, p)| {
                                    *c ^= *p;
                                });

                                let sum = &mut my_sums[keep];

                                // Update the running sums for this level. We keep
                                // a left and right totals for each level.
                                sum.iter_mut().zip(child).for_each(|(s, c)| {
                                    *s ^= *c;
                                });
                                keep += 1;
                                child_idx += 1;
                            }
                        }

                        // For everything but the last level we have to
                        // 1) fix our sums so they dont include the incorrect
                        //    values that are the children of the active parent
                        // 2) Update the non-active child of the active parent.
                        if d != depth - 1 {
                            for i in 0..8 {
                                let leaf_idx = points[i + g];
                                let active_child_idx = leaf_idx >> (depth - 1 - d);
                                let inactive_child_idx = active_child_idx ^ 1;
                                let not_ai = inactive_child_idx & 1;
                                let inactive_child = &mut level1[inactive_child_idx][i];
                                let correct_sum = *inactive_child ^ tree_group.sums[not_ai][d][i];
                                *inactive_child =
                                    correct_sum ^ my_sums[not_ai][i] ^ base_ots[[i + g, d]];
                            }
                        }
                    }
                    // Now processes the last level. This one is special
                    // because we we must XOR in the correction value as
                    // before but we must also fixed the child value for
                    // the active child. To do this, we will receive 4
                    // values. Two for each case (left active or right active).

                    let level = get_level(&mut tree, depth);
                    let d = depth - 1;
                    for j in 0..8 {
                        // The index of the child on the active path.
                        let active_child_idx = points[j + g];
                        // The index of the other (inactive) child.
                        let inactive_child_idx = active_child_idx ^ 1;
                        // The indicator as to the left or right child is inactive
                        let not_ai = inactive_child_idx & 1;

                        // We are going to expand the 128 bit OT string
                        // into a 256 bit OT string using AES.
                        let mask_in = [
                            base_ots[[g + j, d]],
                            base_ots[[g + j, d]] ^ Block::all_ones(),
                        ];
                        let masks = FIXED_KEY_HASH.cr_hash_blocks(&mask_in);

                        // now get the chosen message OT strings by XORing
                        // the expended (random) OT strings with the lastOts values.
                        let ots = [0, 1].map(|i| tree_group.last_ots[j][2 * not_ai + i] ^ masks[i]);

                        // We need to do this little dance as we can't just mutably alias level
                        let children = match active_child_idx.cmp(&inactive_child_idx) {
                            Ordering::Less => {
                                let (left, right) = level.split_at_mut(inactive_child_idx);
                                [&mut right[0], &mut left[active_child_idx]]
                            }
                            Ordering::Greater => {
                                let (left, right) = level.split_at_mut(active_child_idx);
                                [&mut left[inactive_child_idx], &mut right[0]]
                            }
                            Ordering::Equal => unreachable!(
                                "Impossible, active and inactive indices are always different"
                            ),
                        };
                        let [inactive_child, active_child] = children.map(|arr| &mut arr[j]);

                        // Fix the sums we computed previously to not include the
                        // incorrect child values.
                        let inactive_sum = my_sums[not_ai][j] ^ *inactive_child;
                        let active_sum = my_sums[not_ai ^ 1][j] ^ *active_child;
                        *inactive_child = ots[0] ^ inactive_sum;
                        *active_child = ots[1] ^ active_sum;
                    }
                    // copy the last level to the output. If desired, this is
                    // where the tranpose is performed.
                    let last_level = get_level(&mut tree, depth);
                    let mut output = output_clone.lock().unwrap();
                    copy_out(last_level, &mut output, pnt_count, g);
                });
        };

        let par_compute = move || {
            // TODO: this changes the meaning of num_threads. By using par_iter, it becomes the
            // maximum number of threads
            (0..num_threads).into_par_iter().for_each(routine);
        };

        let compute_fut = match thread_pool {
            None => spawn_compute(par_compute),
            Some(pool) => pool.spawn_install_compute(par_compute),
        };
        tokio::join!(distribute_task, compute_fut);

        let output = Arc::try_unwrap(output).unwrap();
        output.into_inner().unwrap()
    }

    // Returns indices of points
    pub fn get_points(&self, format: PprfOutputFormat) -> Vec<usize> {
        match format {
            PprfOutputFormat::Plain => self
                .base_choices
                .rows()
                .into_iter()
                .map(|choice_bits| get_active_path(choice_bits.as_slice().unwrap()))
                .collect(),
            PprfOutputFormat::InterleavedTransposed => {
                let mut points = self.get_points(PprfOutputFormat::Plain);
                interleave_points(&mut points);
                points
            }
        }
    }

    pub fn sample_choice_bits<RNG: RngCore + CryptoRng>(
        conf: PprfConfig,
        modulus: usize,
        format: PprfOutputFormat,
        rng: &mut RNG,
    ) -> ChoiceBits {
        let mut choices = Array2::default((conf.pnt_count, conf.depth));
        for (i, mut choice_row) in choices.rows_mut().into_iter().enumerate() {
            match format {
                PprfOutputFormat::Plain => {
                    let mut idx;
                    loop {
                        choice_row
                            .iter_mut()
                            .for_each(|choice| *choice = rng.gen::<bool>() as u8);
                        idx = get_active_path(choice_row.as_slice().unwrap());
                        if idx < modulus {
                            break;
                        }
                    }
                }
                PprfOutputFormat::InterleavedTransposed => {
                    // make sure that atleast the first element of this tree
                    // is within the modulus.
                    let mut idx = interleave_point(0, i, conf.pnt_count);
                    assert!(idx < modulus, "Iteration {i}, failed: {idx} < {modulus}");
                    loop {
                        choice_row
                            .iter_mut()
                            .for_each(|choice| *choice = rng.gen::<bool>() as u8);
                        idx = get_active_path(choice_row.as_slice().unwrap());
                        idx = interleave_point(idx, i, conf.pnt_count);
                        if idx < modulus {
                            break;
                        }
                    }
                }
            }
        }
        ChoiceBits(choices)
    }
}

impl PprfConfig {
    pub fn new(domain: usize, pnt_count: usize) -> Self {
        let depth = log2_ceil(domain) as usize;
        Self {
            pnt_count,
            domain,
            depth,
        }
    }

    pub fn base_ot_count(&self) -> usize {
        self.depth * self.pnt_count
    }

    pub fn pnt_count(&self) -> usize {
        self.pnt_count
    }

    pub fn domain(&self) -> usize {
        self.domain
    }

    pub fn depth(&self) -> usize {
        self.depth
    }
}

impl ChoiceBits {
    pub fn as_bit_vec(&self) -> BitVec {
        BitVec::from_iter(self.iter())
    }

    pub fn iter(&self) -> impl Iterator<Item = bool> + '_ {
        self.0.iter().map(|bit| *bit != 0)
    }
}

// Create a pair of fixed key aes128 ciphers
fn create_fixed_aes() -> [Aes128; 2] {
    [
        Aes128::new(
            &91389970179024809574621370423327856399_u128
                .to_le_bytes()
                .into(),
        ),
        Aes128::new(
            &297966570818470707816499469807199042980_u128
                .to_le_bytes()
                .into(),
        ),
    ]
}

// Todo: choice_bits contains bits as individual u8, we can probably
//  refactor this to use bitvec
fn get_active_path(choice_bits: &[u8]) -> usize {
    choice_bits.iter().enumerate().fold(0, |point, (i, &cb)| {
        let shift = choice_bits.len() - i - 1;
        point | ((1 ^ cb as usize) << shift)
    })
}

fn interleave_points(points: &mut [usize]) {
    let total_trees = points.len();
    points
        .iter_mut()
        .enumerate()
        .for_each(|(i, point)| *point = interleave_point(*point, i, total_trees))
}

fn interleave_point(point: usize, tree_idx: usize, total_trees: usize) -> usize {
    let num_sets = total_trees / 8;

    let set_idx = tree_idx / 8;
    let sub_idx = tree_idx % 8;

    let section_idx = point / 16;
    let pos_idx = point % 16;

    let set_offset = set_idx * 128;
    let sub_offset = sub_idx + 8 * pos_idx;
    let sec_offset = section_idx * num_sets * 128;

    set_offset + sub_offset + sec_offset
}

#[derive(Serialize, Deserialize, Default, Clone, Debug)]
pub struct TreeGrp {
    g: usize,
    sums: [Vec<[Block; 8]>; 2],
    last_ots: Vec<[Block; 4]>,
}
#[derive(Error, Debug)]
pub enum ExpandError {
    #[error("unknown error")]
    Unknown,
}

// Returns the i'th level of the current 8 trees. The
// children of node j on level i are located at 2*j and
// 2*j+1  on level i+1.
fn get_level(tree: &mut [[Block; 8]], i: usize) -> &mut [[Block; 8]] {
    let size = 1 << i;
    let offset = size - 1;
    &mut tree[offset..offset + size]
}

// Return the i'th and (i+1)'th level
fn get_cons_levels(tree: &mut [[Block; 8]], i: usize) -> (&mut [[Block; 8]], &mut [[Block; 8]]) {
    let size0 = 1 << i;
    let offset0 = size0 - 1;
    let tree = &mut tree[offset0..];
    let (level0, rest) = tree.split_at_mut(size0);
    let size1 = 1 << (i + 1);
    debug_assert_eq!(size0 + offset0, size1 - 1);
    let level1 = &mut rest[..size1];
    (level0, level1)
}

fn copy_out(lvl: &[[Block; 8]], output: &mut Array2<Block>, total_trees: usize, t_idx: usize) {
    assert_eq!(total_trees % 8, 0, "Number of trees must be dividable by 8");
    assert_eq!(lvl.len() % 16, 0, "lvl len() must be dividable by 16");
    assert!(lvl.len() > 16, "Lvl must have size of at least 16");

    let set_idx = t_idx / 8;
    let block_per_set = lvl.len() * 8 / 128;
    let num_sets = total_trees / 8;
    let step = num_sets;
    let end = cmp::min(set_idx + step * block_per_set, output.ncols());
    let mut i = set_idx;
    let mut k = 0;
    while i < end {
        // get 128 blocks
        let input_128: &[u8] = bytemuck::cast_slice(&lvl[k * 16..(k + 1) * 16]);
        let transposed = transpose(input_128, 128, 128);
        let transposed_blocks = transposed
            .chunks_exact(16)
            .map(|chunk| Block::try_from(chunk).expect("Blocks are 16 bytes"));
        output
            .rows_mut()
            .into_iter()
            .zip(transposed_blocks)
            .for_each(|(mut row, block)| {
                row[i] = block;
            });
        i += step;
        k += 1;
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::silent_ot::pprf::{ChoiceBits, PprfConfig, PprfOutputFormat, Receiver, Sender};
    use crate::util::transpose::transpose;
    use crate::util::Block;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use rand_core::{CryptoRng, RngCore};
    use tokio::time::Instant;

    pub(crate) fn fake_base<RNG: RngCore + CryptoRng>(
        pprf_conf: PprfConfig,
        modulus: usize,
        format: PprfOutputFormat,
        rng: &mut RNG,
    ) -> (Vec<[Block; 2]>, Vec<Block>, ChoiceBits) {
        let base_ot_count = pprf_conf.base_ot_count();
        let msg2: Vec<[Block; 2]> = (0..base_ot_count).map(|_| rng.gen()).collect();
        let choices = Receiver::sample_choice_bits(pprf_conf, modulus, format, rng);
        let msg = msg2
            .iter()
            .zip(choices.iter())
            .map(|(m, c)| m[c as usize])
            .collect();
        (msg2, msg, choices)
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn silent_pprf() {
        let now = Instant::now();
        let conf = PprfConfig::new(334, 5 * 8);
        let format = PprfOutputFormat::InterleavedTransposed;
        let mut rng = StdRng::seed_from_u64(42);

        let threads = 1;
        let ((sender_ch, _), (_, receiver_ch)) = mpc_channel::in_memory::new_pair(128);
        let (sender_base_ots, receiver_base_ots, base_choices) =
            fake_base(conf, conf.domain * conf.pnt_count, format, &mut rng);
        let send_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .into();
        let recv_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .into();

        let send = tokio::spawn(async move {
            let mut sender = Sender::new(conf, sender_base_ots);
            sender
                .expand(sender_ch, Block::all_ones(), &mut rng, Some(send_pool))
                .await
        });
        let mut receiver = Receiver::new(conf, receiver_base_ots, base_choices);
        let points = receiver.get_points(format);
        let receive =
            tokio::spawn(async move { receiver.expand(receiver_ch, Some(recv_pool)).await });
        let (r_out, s_out) = futures::future::try_join(receive, send).await.unwrap();
        println!("Total time: {}", now.elapsed().as_secs_f32());
        let out = r_out ^ s_out;
        let out_t = transpose(
            bytemuck::cast_slice(out.as_slice().unwrap()),
            out.nrows(),
            out.ncols() * 128, // * 128 because of Block size
        );
        let out_t: &[Block] = bytemuck::cast_slice(&out_t);
        for (i, blk) in (&out_t[0..conf.domain * conf.pnt_count]).iter().enumerate() {
            let f = points.contains(&i);
            let exp = if f { Block::all_ones() } else { Block::zero() };
            assert_eq!(*blk, exp, "block {i} not as expected");
        }
    }
}
