use crate::utils::RangeInclusiveStartWrapper;
use crate::{bristol, SubCircuitGate};
pub use builder::SubCircCache;
use num_integer::Integer;
use petgraph::adj::IndexType;
use smallvec::SmallVec;
use std::collections::{BTreeMap, Bound, HashMap};
use std::fmt::Debug;
use std::hash::Hash;
use std::ops::RangeInclusive;
use std::path::Path;
use std::sync::Arc;
use tracing::trace;

pub mod base_circuit;
pub mod builder;

use crate::circuit::base_circuit::{BaseGate, Load};
use crate::errors::CircuitError;
pub use crate::protocols::boolean_gmw::BooleanGate;
use crate::protocols::Gate;
pub use base_circuit::{BaseCircuit, GateId};
pub use builder::{CircuitBuilder, SharedCircuit};

pub type CircuitId = u32;

pub trait GateIdx:
    IndexType
    + Integer
    + Copy
    + Send
    + Sync
    + TryFrom<usize>
    + TryFrom<u32>
    + TryFrom<u16>
    + Into<GateId<Self>>
{
}

impl<
        T: IndexType
            + Integer
            + Copy
            + Send
            + Sync
            + TryFrom<usize>
            + TryFrom<u32>
            + TryFrom<u16>
            + Into<GateId<Self>>,
    > GateIdx for T
{
}

pub type DefaultIdx = u32;

pub trait LayerIterable {
    type Layer;
    type LayerIter<'this>: Iterator<Item = Self::Layer>
    where
        Self: 'this;

    fn layer_iter(&self) -> Self::LayerIter<'_>;
}

#[derive(Debug, Clone)]
pub struct Circuit<G = BooleanGate, Idx = DefaultIdx> {
    pub(crate) circuits: Vec<Arc<BaseCircuit<G, Idx>>>,
    pub(crate) connections: CrossCircuitConnections<Idx>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct CrossCircuitConnections<Idx> {
    pub(crate) one_to_one: OneToOneConnections<Idx>,
    pub(crate) range_connections: RangeConnections<Idx>,
}

type OneToOneMap<Idx, const BUF_SIZE: usize> =
    HashMap<SubCircuitGate<Idx>, SmallVec<[SubCircuitGate<Idx>; BUF_SIZE]>>;

#[derive(Debug, Clone, Default)]
pub(crate) struct OneToOneConnections<Idx> {
    pub(crate) outgoing: OneToOneMap<Idx, 1>,
    pub(crate) incoming: OneToOneMap<Idx, 2>,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct RangeConnections<Idx> {
    pub(crate) incoming: RangeSubCircuitConnections<Idx>,
    pub(crate) outgoing: RangeSubCircuitConnections<Idx>,
}

type FromRange<Idx> = RangeInclusiveStartWrapper<SubCircuitGate<Idx>>;
type ToRanges<Idx> = SmallVec<[RangeInclusive<SubCircuitGate<Idx>>; 1]>;

#[derive(Debug, Clone, Default)]
pub struct RangeSubCircuitConnections<Idx> {
    map: HashMap<CircuitId, BTreeMap<FromRange<Idx>, ToRanges<Idx>>>,
}

impl<Idx: GateIdx> CrossCircuitConnections<Idx> {
    fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.one_to_one
            .parent_gates(id)
            .chain(self.range_connections.parent_gates(id))
    }

    fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.one_to_one
            .outgoing_gates(id)
            .chain(self.range_connections.outgoing_gates(id))
    }
}

impl<Idx: GateIdx> OneToOneConnections<Idx> {
    fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.incoming
            .get(&id)
            .map(SmallVec::as_slice)
            .unwrap_or(&[])
            .iter()
            .copied()
    }

    fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        self.outgoing
            .get(&id)
            .map(|sv| sv.iter().copied())
            .into_iter()
            .flatten()
    }
}

impl<Idx> RangeConnections<Idx>
where
    Idx: GateIdx,
{
    fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let range_conns = self.incoming.get_mapped_ranges(id);
        range_conns.flat_map(move |(to_range, from_ranges)| {
            if !to_range.contains(&id) {
                unreachable!("to_range is wrong");
            }
            let offset = id.gate_id.0 - to_range.start().gate_id.0;

            from_ranges.iter().map(move |from_range| {
                let from_range_start = from_range.start();
                let from_gate_id = (from_range_start.gate_id.0 + offset).into();
                SubCircuitGate::new(from_range_start.circuit_id, from_gate_id)
            })
        })
    }

    #[tracing::instrument(level = "trace", skip(self))]
    fn outgoing_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let range_conns = self.outgoing.get_mapped_ranges(id);
        range_conns.flat_map(move |(from_range, to_ranges)| {
            trace!(?from_range, ?to_ranges, "range_conns_outgoing_gates");
            if !from_range.contains(&id) {
                unreachable!("from_range does not contain id")
            }
            let offset = id.gate_id.0 - from_range.start().gate_id.0;
            to_ranges.iter().map(move |to_range| {
                let to_gate_id = (to_range.start().gate_id.0 + offset).into();
                SubCircuitGate::new(to_range.start().circuit_id, to_gate_id)
            })
        })
    }
}

impl<G: Gate, Idx: GateIdx> Circuit<G, Idx> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_circuit(&mut self, circuit: impl Into<Arc<BaseCircuit<G, Idx>>>) {
        self.circuits.push(circuit.into());
    }

    // TODO optimization!
    pub fn parent_gates(
        &self,
        id: SubCircuitGate<Idx>,
    ) -> impl Iterator<Item = SubCircuitGate<Idx>> + '_ {
        let same_circuit = self.circuits[id.circuit_id as usize]
            .parent_gates(id.gate_id)
            .map(move |parent_gate| SubCircuitGate::new(id.circuit_id, parent_gate));

        same_circuit.chain(self.connections.parent_gates(id))
    }

    pub fn gate_count(&self) -> usize {
        self.circuits.iter().map(|circ| circ.gate_count()).sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        // Reuse the the CircuitLayerIter api to get an iterator over individual gates. This
        // is maybe a little more inefficient than necessary, but probably fine for the moment,
        // as this method is expected to be called in the preprocessing phase
        let layer_iter = CircuitLayerIter::new(self);
        layer_iter.flat_map(|layer| {
            layer.sc_layers.into_iter().flat_map(|(sc_id, base_layer)| {
                base_layer
                    .non_interactive
                    .into_iter()
                    .map(move |(gate, gate_id)| (gate, SubCircuitGate::new(sc_id, gate_id)))
                    // Interactive gates need be chained after interactive gates
                    .chain(
                        base_layer
                            .interactive_gates
                            .into_iter()
                            .map(move |(gate, gate_id)| {
                                (gate, SubCircuitGate::new(sc_id, gate_id))
                            }),
                    )
            })
        })
    }

    pub fn interactive_iter(&self) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        // TODO this can be optimized in the future
        self.iter().filter(|(gate, _)| gate.is_interactive())
    }

    // pub fn into_base_circuit(self) -> BaseCircuit<Idx> {
    //     let mut res = BaseCircuit::new();
    //     let mut new_ids: Vec<Vec<_>> = vec![];
    //     for (c_id, circ) in self.circuits.into_iter().enumerate() {
    //         let g = circ.as_graph();
    //         let (nodes, edges) = (g.raw_nodes(), g.raw_edges());
    //
    //         new_ids.push(nodes.into_iter().map(|n| res.add_gate(n.weight)).collect());
    //         for edge in edges {
    //             let from = new_ids[c_id][edge.source().index()];
    //             let to = new_ids[c_id][edge.target().index()];
    //             res.add_wire(from, to);
    //         }
    //         // TODO DRY
    //         // TODO to convert this to the map implementation, maybe not iterate over input gates
    //         //  of circ but iterate over all edges in self.circuit_connectioctions at the end
    //         //  to link up circuits
    //         for input_gate in circ.sub_circuit_input_gates() {
    //             let to = (c_id.try_into().unwrap(), *input_gate);
    //             for from in self
    //                 .circuit_connections
    //                 .neighbors_directed(to, Direction::Incoming)
    //             {
    //                 if from.0 as usize >= new_ids.len() {
    //                     continue;
    //                 }
    //                 let from = new_ids[from.0 as usize][from.1.as_usize()];
    //                 res.add_wire(from, new_ids[c_id][to.1.as_usize()]);
    //             }
    //         }
    //         // TODO remove following when removing interleaving of SCs
    //         for output_gate in circ.sub_circuit_output_gates() {
    //             let from = (c_id.try_into().unwrap(), *output_gate);
    //             for to in self
    //                 .circuit_connections
    //                 .neighbors_directed(from, Direction::Outgoing)
    //             {
    //                 if to.0 as usize >= new_ids.len() {
    //                     continue;
    //                 }
    //                 let from = new_ids[from.0 as usize][from.1.as_usize()];
    //                 res.add_wire(from, new_ids[to.0 as usize][to.1.as_usize()]);
    //             }
    //         }
    //     }
    //
    //     res
    // }
}

impl<G, Idx> Circuit<G, Idx> {
    pub fn interactive_count(&self) -> usize {
        self.circuits
            .iter()
            .map(|circ| circ.interactive_count())
            .sum()
    }

    /// Returns the input count of the **main circuit**.
    pub fn input_count(&self) -> usize {
        self.circuits[0].input_count()
    }

    /// Returns the output count of the **main circuit**.
    pub fn output_count(&self) -> usize {
        self.circuits[0].output_count()
    }
}

impl<Share, G> Circuit<G, usize>
where
    Share: Copy,
    G: Gate<Share = Share> + From<BaseGate<Share>> + for<'a> From<&'a bristol::Gate>,
{
    pub fn load_bristol(path: impl AsRef<Path>) -> Result<Self, CircuitError> {
        BaseCircuit::load_bristol(path, Load::Circuit).map(Into::into)
    }
}

impl<Idx: Ord + Copy + IndexType + Debug> RangeSubCircuitConnections<Idx> {
    /// # Panics
    /// This function asserts that the from range is not a strictly contained in an
    /// already stored range.
    /// E.g. When (3..=9) is stored, it is illegal to store (4..=6)
    /// It **is** allowed, to store ranges with the same start but differing lengths, e.g. (3..=5)
    pub(crate) fn insert(
        &mut self,
        from: RangeInclusive<SubCircuitGate<Idx>>,
        to: RangeInclusive<SubCircuitGate<Idx>>,
    ) {
        assert!(
            from.start() <= from.end(),
            "from.start() must be <= than end()"
        );
        let from_circuit_id = from.start().circuit_id;
        let new_from_start_wrapper = RangeInclusiveStartWrapper::new(from.clone());
        let bmap = self.map.entry(from_circuit_id).or_default();
        let potential_conflict = bmap
            .range((
                Bound::Unbounded,
                Bound::Included(new_from_start_wrapper.clone()),
            ))
            .next_back();
        if let Some((potential_conflict, _)) = potential_conflict {
            assert!(
                !(potential_conflict.range.start() < from.start()
                    && from.end() <= potential_conflict.range.end()),
                "RangeSubCircuitConnections can't store a range which is a \
                strict sub range of an already stored one"
            );
        }

        bmap.entry(new_from_start_wrapper).or_default().push(to);
    }

    // TODO is it possible to provide an API that takes multiple (sorted?) gates and more
    //  efficiently returns the mapped ranges?
    pub(crate) fn get_mapped_ranges(
        &self,
        gate: SubCircuitGate<Idx>,
    ) -> impl Iterator<
        Item = (
            RangeInclusive<SubCircuitGate<Idx>>,
            &[RangeInclusive<SubCircuitGate<Idx>>],
        ),
    > {
        let start_wrapper = RangeInclusiveStartWrapper::new(
            gate..=SubCircuitGate::new(gate.circuit_id, GateId(<Idx as IndexType>::max())),
        );
        self.map
            .get(&gate.circuit_id)
            .into_iter()
            .flat_map(move |bmap| {
                bmap.range((Bound::Unbounded, Bound::Included(start_wrapper.clone())))
                    .rev()
                    .take_while(move |(range_wrapper, _to_ranges)| {
                        *range_wrapper.range.start() <= gate && gate <= *range_wrapper.range.end()
                    })
                    .map(|(from_range_wrapper, to_vec)| {
                        (from_range_wrapper.range.clone(), to_vec.as_slice())
                    })
            })
    }
}

#[derive(Clone)]
pub struct CircuitLayerIter<'a, G, Idx: GateIdx> {
    circuit: &'a Circuit<G, Idx>,
    layer_iters: HashMap<CircuitId, base_circuit::BaseLayerIter<'a, G, Idx>>,
}

impl<'a, G: Gate, Idx: GateIdx> CircuitLayerIter<'a, G, Idx> {
    pub fn new(circuit: &'a Circuit<G, Idx>) -> Self {
        let first_iter = circuit.circuits[0].layer_iter();
        Self {
            circuit,
            layer_iters: [(0, first_iter)].into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CircuitLayer<G, Idx: Hash + PartialEq + Eq> {
    sc_layers: Vec<(CircuitId, base_circuit::CircuitLayer<G, Idx>)>,
}

impl<'a, G: Gate, Idx: GateIdx> Iterator for CircuitLayerIter<'a, G, Idx> {
    type Item = CircuitLayer<G, Idx>;

    // TODO optimize this method, it makes up a big part of the runtime
    // TODO remove BaseLayerIters when they are not use anymore
    fn next(&mut self) -> Option<Self::Item> {
        // let now = Instant::now();
        trace!("layer_iters: {:#?}", &self.layer_iters);
        let mut sc_layers: Vec<_> = self
            .layer_iters
            .iter_mut()
            .filter_map(|(&sc_id, iter)| iter.next().map(|layer| (sc_id, layer)))
            .collect();
        // Only retain iters which can yield elements
        // TODO: this code is buggy, as the base iters are not necessarily fused, it could happen
        //  that a base iter yields none because it waits for dependencies byt later yields Some
        // self.layer_iters.retain(|&sc_id, iter| match iter.next() {
        //     None => false,
        //     Some(layer) => {
        //         sc_layers.push((sc_id, layer));
        //         true
        //     }
        // });

        // This is crucial as the executor depends on the and gates being in the same order for
        // both parties
        sc_layers.sort_unstable_by_key(|sc_id| sc_id.0);
        for (sc_id, layer) in &sc_layers {
            // TODO this iterates over all range connections for circuit in sc_layers for every
            //  layer. This could slow down the layer generation. Use hashmaps?
            for (_, potential_out) in layer.non_interactive.iter().chain(&layer.interactive_gates) {
                let from = SubCircuitGate::new(*sc_id, *potential_out);
                let outgoing = self.circuit.connections.outgoing_gates(from);

                for sc_gate in outgoing {
                    let to_layer_iter =
                        self.layer_iters
                            .entry(sc_gate.circuit_id)
                            .or_insert_with(|| {
                                base_circuit::BaseLayerIter::new_uninit(
                                    &self.circuit.circuits[sc_gate.circuit_id as usize],
                                )
                            });
                    to_layer_iter.add_to_next_layer(sc_gate.gate_id.into());
                }
            }
        }
        // println!("next took {}", now.elapsed().as_secs_f32());
        // Todo can there be circuits where a layer is empty, but a later call to next returns
        //  a layer?
        if sc_layers.is_empty() {
            None
        } else {
            Some(CircuitLayer { sc_layers })
        }
    }
}

impl<G: Gate, Idx: Hash + PartialEq + Eq + Copy> CircuitLayer<G, Idx> {
    pub(crate) fn non_interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + '_ {
        self.sc_layers.iter().flat_map(|(sc_id, layer)| {
            layer
                .non_interactive
                .iter()
                .cloned()
                .map(|(gate, gate_idx)| (gate, SubCircuitGate::new(*sc_id, gate_idx)))
        })
    }

    pub(crate) fn interactive_iter(
        &self,
    ) -> impl Iterator<Item = (G, SubCircuitGate<Idx>)> + Clone + '_ {
        self.sc_layers.iter().flat_map(|(sc_id, layer)| {
            layer
                .interactive_gates
                .iter()
                .cloned()
                .map(|(gate, gate_idx)| (gate, SubCircuitGate::new(*sc_id, gate_idx)))
        })
    }
}

impl<G, Idx: GateIdx> Default for Circuit<G, Idx> {
    fn default() -> Self {
        Self {
            circuits: vec![],
            connections: Default::default(),
        }
    }
}

impl<G, Idx: GateIdx + Default> From<BaseCircuit<G, Idx>> for Circuit<G, Idx> {
    fn from(bc: BaseCircuit<G, Idx>) -> Self {
        Self {
            circuits: vec![Arc::new(bc)],
            ..Default::default()
        }
    }
}

impl<G, Idx: GateIdx> TryFrom<SharedCircuit<G, Idx>> for Circuit<G, Idx> {
    type Error = SharedCircuit<G, Idx>;

    fn try_from(circuit: SharedCircuit<G, Idx>) -> Result<Self, Self::Error> {
        Arc::try_unwrap(circuit).map(|mutex| mutex.into_inner().into())
    }
}

#[cfg(test)]
mod tests {
    use crate::circuit::RangeSubCircuitConnections;
    use crate::{GateId, SubCircuitGate};

    #[test]
    fn test_range_connections_simple() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let to_range =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));

        rc.insert(from_range.clone(), to_range.clone());

        for (ret_from_range, ret_to_ranges) in
            rc.get_mapped_ranges(SubCircuitGate::new(0, GateId(0_u32)))
        {
            assert_eq!(ret_from_range, from_range);
            assert_eq!(ret_to_ranges[0], to_range);
            assert_eq!(ret_to_ranges.len(), 1);
        }
    }

    #[test]
    fn test_range_connections_overlapping() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(50_u32))..=SubCircuitGate::new(0, GateId(100_u32));
        let to_range =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));

        rc.insert(from_range_0.clone(), to_range.clone());
        rc.insert(from_range_1.clone(), to_range.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(50_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(mapped_ranges[1], (from_range_0, &[to_range.clone()][..]));
        assert_eq!(mapped_ranges[0], (from_range_1, &[to_range][..]));
    }

    #[test]
    fn test_range_connections_inside_each_other_same_start() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(20_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(120_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_1.clone(), to_range_1.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(20_u32)))
            .collect();
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(mapped_ranges[0], (from_range_0, &[to_range_0.clone()][..]));
        assert_eq!(mapped_ranges[1], (from_range_1, &[to_range_1][..]));
    }

    #[test]
    fn test_range_connections_map_to_multiple() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(2, GateId(100_u32))..=SubCircuitGate::new(2, GateId(150_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_0.clone(), to_range_1.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(20_u32)))
            .collect();
        assert_eq!(mapped_ranges.len(), 1);
        assert_eq!(
            mapped_ranges[0],
            (from_range_0, &[to_range_0, to_range_1][..])
        );
    }

    #[test]
    fn test_range_connections_regression() {
        let mut rc = RangeSubCircuitConnections::default();

        let from_range_0 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(8_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(167_u32),
        };
        let to_range_0 = SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(159_u32),
        };
        rc.insert(from_range_0.clone(), to_range_0.clone());
        let from_range_1 = SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(280_u32),
        }..=SubCircuitGate {
            circuit_id: 1,
            gate_id: GateId(339_u32),
        };
        let to_range_1 = SubCircuitGate {
            circuit_id: 2,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 2,
            gate_id: GateId(59_u32),
        };
        rc.insert(from_range_1.clone(), to_range_1.clone());
        let from_range_2 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(8_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(87_u32),
        };
        let to_range_2 = SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(0_u32),
        }..=SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(79_u32),
        };
        rc.insert(from_range_2.clone(), to_range_2.clone());
        let from_range_4 = SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(96_u32),
        }..=SubCircuitGate {
            circuit_id: 0,
            gate_id: GateId(175_u32),
        };
        let to_range_4 = SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(80_u32),
        }..=SubCircuitGate {
            circuit_id: 3,
            gate_id: GateId(159_u32),
        };
        rc.insert(from_range_4.clone(), to_range_4.clone());

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(87_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 2);
        assert_eq!(
            mapped_ranges[0],
            (from_range_0.clone(), &[to_range_0.clone()][..])
        );
        assert_eq!(mapped_ranges[1], (from_range_2, &[to_range_2][..]));

        let mapped_ranges: Vec<_> = rc
            .get_mapped_ranges(SubCircuitGate::new(0, GateId(88_u32)))
            .collect();
        dbg!(&rc.map);
        dbg!(&mapped_ranges);
        assert_eq!(mapped_ranges.len(), 1);
        assert_eq!(mapped_ranges[0], (from_range_0, &[to_range_0][..]));
    }

    #[test]
    #[should_panic]
    fn test_range_connections_inside_each_other_illegal() {
        let mut rc = RangeSubCircuitConnections::default();
        let from_range_0 =
            SubCircuitGate::new(0, GateId(0_u32))..=SubCircuitGate::new(0, GateId(50_u32));
        let from_range_1 =
            SubCircuitGate::new(0, GateId(10_u32))..=SubCircuitGate::new(0, GateId(20_u32));
        let to_range_0 =
            SubCircuitGate::new(1, GateId(100_u32))..=SubCircuitGate::new(1, GateId(150_u32));
        let to_range_1 =
            SubCircuitGate::new(1, GateId(110_u32))..=SubCircuitGate::new(1, GateId(120_u32));

        rc.insert(from_range_0.clone(), to_range_0.clone());
        rc.insert(from_range_1.clone(), to_range_1.clone());
    }
}
