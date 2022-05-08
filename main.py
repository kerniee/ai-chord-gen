import math
import random
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from itertools import combinations, chain
from typing import List, Optional
from typing import Tuple, Callable, TypeVar, Set, FrozenSet, Any

from mido import MidiFile, MidiTrack, Message, MetaMessage
try:
    from tqdm import trange
except ImportError as e:
    trange = range


class PitchClass(Enum):
    C = 0
    CS = 1
    D = 2
    DS = 3
    E = 4
    F = 5
    FS = 6
    G = 7
    GS = 8
    A = 9
    AS = 10
    B = 11

    @property
    def name(self):
        return self._name_.replace("S", "#")

    def __add__(self, other):
        if isinstance(other, PitchClass):
            return self.__class__((self.value + other.value) % 12)
        else:
            return self.__class__((self.value + other) % 12)


def generate_pitch_to_num() -> dict:
    res = dict()
    for i in range(128):
        pc = PitchClass(i % 12)
        octave = i // 12
        pitch = f"{pc.name}{octave}"
        res[pitch] = i
    return res


Pitch = Enum("Pitch", generate_pitch_to_num(), type=int)


@dataclass
class Note:
    pitch: Pitch
    duration: float = 1
    velocity: int = 100


@dataclass
class Scale:
    scale_intervals: List[int]
    chords: Callable[[PitchClass, List[int]], Set[FrozenSet[PitchClass]]]
    name: Optional[str] = None

    def pitch_classes(self, root_pc: PitchClass):
        value = root_pc.value
        return frozenset({PitchClass((value + x) % 12) for x in self.scale_intervals})

    @lru_cache
    def get_chords(self, root_pc: PitchClass) -> Set[FrozenSet[PitchClass]]:
        """
        Possible chords for such scale
        :param root_pc: root note
        :return: list of chords, where chord is a list of notes
        """
        return self.chords(root_pc, self.scale_intervals)

    def __hash__(self):
        return hash(tuple(self.scale_intervals))


def possible_chords(root_pc, scale_intervals, allowed_chords, sus2_forbidden, sus4__forbidden):
    sus2 = [0, 2, 7]
    sus4 = [0, 5, 7]

    def chord_from_offsets(root, offsets):
        return frozenset({root + offset for offset in offsets})

    ans = set()
    for i, interval in enumerate(scale_intervals):
        current_note = root_pc + interval
        ans.add(chord_from_offsets(current_note, allowed_chords[i]))
        if i + 1 not in sus2_forbidden:
            ans.add(chord_from_offsets(current_note, sus2))
        if i + 1 not in sus4__forbidden:
            ans.add(chord_from_offsets(current_note, sus4))
    return ans


def possible_chords_major_scale(root_pc: PitchClass, scale_intervals: List[int]) -> Set[FrozenSet[PitchClass]]:
    # integer notation chords
    major = [0, 4, 7]
    minor = [0, 3, 7]
    dim = [0, 3, 6]

    allowed_chords = [
        major,
        minor,
        minor,
        major,
        major,
        minor,
        dim
    ]
    return possible_chords(root_pc, scale_intervals, allowed_chords,
                           sus2_forbidden=[3, 7], sus4__forbidden=[4, 7])


def possible_chords_minor_scale(root_pc: PitchClass, scale_intervals: List[int]) -> set[frozenset[Any]]:
    # integer notation chords
    major = [0, 4, 7]
    minor = [0, 3, 7]
    dim = [0, 3, 6]

    allowed_chords = [
        minor,
        dim,
        major,
        minor,
        minor,
        major,
        major
    ]
    return possible_chords(root_pc, scale_intervals, allowed_chords,
                           sus2_forbidden=[2, 5], sus4__forbidden=[2, 6])


major_scale = Scale([0, 2, 4, 5, 7, 9, 11], name="Major",
                    chords=possible_chords_major_scale)
minor_scale = Scale([0, 2, 3, 5, 7, 8, 10], name="Natural minor",
                    chords=possible_chords_minor_scale)


class Midi:
    @staticmethod
    def track_to_notes(track: MidiTrack, tick_per_beat):
        notes = []
        velocity = 0
        note_on_mes_c = 0
        duration_total = 0
        for message in track:
            duration = message.time / tick_per_beat
            if message.type == "note_off":
                notes[-1].duration += duration
            if message.type == "note_on":
                pitch = Pitch(message.note)
                notes.append(Note(pitch, duration))

                velocity += message.velocity
                note_on_mes_c += 1
            duration_total += duration
        return notes, velocity / note_on_mes_c, duration_total

    @staticmethod
    def chords_to_track(chords: List[List[Note]], tick_per_beat, velocity=None) -> MidiTrack:
        track = MidiTrack()

        def note_on(_chord):
            for _note in _chord:
                track.append(
                    Message('note_on', note=_note.pitch, velocity=_note.velocity if velocity is None else velocity,
                            time=0))

        def note_off(_chord, _chord_duration):
            for i, _note in enumerate(_chord):
                track.append(
                    Message('note_off', note=_note.pitch, velocity=_note.velocity if velocity is None else velocity,
                            time=_chord_duration * tick_per_beat if i == 0 else 0))

        last_chord = None
        chord_duration = 0
        for chord in chords:
            if last_chord is None:
                # first chord
                chord_duration += chord[0].duration
                note_on(chord)
            elif last_chord == chord:
                # chord repeated
                chord_duration += chord[0].duration
            else:
                # new chord
                note_off(last_chord, chord_duration)

                chord_duration = chord[0].duration
                note_on(chord)
            last_chord = chord

            # assert duration of the notes in chord is the same
            assert len({note.duration for note in chord}) == 1

        track.append(MetaMessage("end_of_track", time=0))
        return track


def possible_scales(notes: List[Note], scales: List[Scale]):
    scales_with_root = set()
    pcs_to_scale_and_root = defaultdict(list)
    for scale in scales:
        for pc in PitchClass:
            pcs = scale.pitch_classes(root_pc=pc)
            pcs_to_scale_and_root[pcs].append((scale, pc))
            scales_with_root.add(pcs)

    pcs_in_notes = set()
    for note in notes:
        pcs_in_notes.add(PitchClass(note.pitch.value % 12))

    ans_pcs = set()
    for pcs in scales_with_root:
        if pcs.issubset(pcs_in_notes) or pcs_in_notes.issubset(pcs):
            ans_pcs.add(pcs)

    ans_scales_and_roots = []
    for pcs in ans_pcs:
        ans_scales_and_roots.extend(pcs_to_scale_and_root[pcs])
    return ans_pcs, ans_scales_and_roots


# ==================================================================== #
# Genetic algorithm base

TGene = TypeVar("TGene", bound="Gene")


class Gene(ABC):
    @abstractmethod
    def mutate(self) -> TGene:
        pass


TChromosome = TypeVar("TChromosome", bound="Chromosome")


class Chromosome:
    def __init__(self, genes: List[Gene]):
        self.genes = genes

    def mutate(self, scale: Scale = None, mutation_chance=0.5) -> TChromosome:
        new_genes = []
        for gene in self.genes:
            if random.random() < mutation_chance:
                new_genes.append(gene.mutate())
            else:
                new_genes.append(gene)
        # swap random genes
        for _ in range(random.randrange(4)):
            i = random.randrange(len(new_genes))
            j = random.randrange(len(new_genes))
            new_genes[i], new_genes[j] = new_genes[j], new_genes[i]
        return Chromosome(new_genes)

    def crossover(self, other: TChromosome) -> Tuple[TChromosome, TChromosome]:
        border = random.randrange(len(self.genes))
        genes_c1 = self.genes[:border]
        genes_c1.extend(other.genes[border:])
        genes_c2 = other.genes[:border]
        genes_c2.extend(self.genes[border:])
        return Chromosome(genes_c1), Chromosome(genes_c2)

    @classmethod
    def generate(cls, genes_size: int, gene_generate: Callable[..., Gene]) -> TChromosome:
        return cls([gene_generate() for _ in range(genes_size)])


class GA:
    def __init__(self, fitness: Callable[[Chromosome], float],
                 gene_generate: Callable[..., Gene],
                 selection: Callable[[List[Chromosome], List[float]], Chromosome] = None):
        self._fit = fitness
        self._gene_generate = gene_generate
        self._best = {"chromosome": None,
                      "fitness": -10 ** 10}
        if selection is not None:
            self._selection = selection
        else:
            self._selection = GA.tournament_selection

    @staticmethod
    def tournament_selection(population, fitnesses, k=3) -> Chromosome:
        selected_ix = random.randrange(len(population))
        for i in range(k - 1):
            ix = random.randrange(len(population))
            if fitnesses[ix] > fitnesses[selected_ix]:
                selected_ix = ix
        return population[selected_ix]

    def run(self, pop_size: int, genes_size, mutation_chance=0.5, iters=10) -> dict:
        population = [Chromosome.generate(genes_size, self._gene_generate) for _ in range(pop_size)]

        for _ in trange(iters):
            fitnesses = [self._fit(chromo) for chromo in population]

            # update best
            for i, fitness in enumerate(fitnesses):
                if fitness > self._best["fitness"]:
                    self._best["fitness"] = fitness
                    self._best["chromosome"] = population[i]

            selected = [self._selection(population, fitnesses) for _ in range(pop_size)]
            children = list()
            for i in range(0, pop_size, 2):
                p1 = selected[i]
                p2 = selected[i + 1]
                for j, c in enumerate(p1.crossover(p2)):
                    if random.random() < mutation_chance:
                        c = c.mutate()
                    children.append(c)
            population = children
        return self._best


# ==================================================================== #
# Custom genetic algorithm things


class ChordGene(Gene):
    possible_chords = None

    def __init__(self, *pitches: PitchClass):
        self.pitches = pitches

    def mutate(self):
        # choices = (-2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2)
        # choices = (0,)
        #
        # return ChordGene(*[p + random.choice(choices) for p in self.pitches])
        return ChordGene.generate()

    @staticmethod
    def generate():
        return ChordGene(*list(random.choice(ChordGene.possible_chords)))


def chord_fit(melody: List[Note]) -> Callable[[Chromosome], float]:
    scales, scales_with_roots = possible_scales(melody, [major_scale, minor_scale])
    bar_to_notes = defaultdict(set)
    duration = 0
    for note in melody:
        bar_to_notes[math.floor(duration)].add(
            PitchClass(note.pitch.value % 12)
        )
        duration += note.duration
    bar_to_notes = defaultdict(frozenset, {k: frozenset(v) for k, v in bar_to_notes.items()})

    def fit_to_scale(genes: List[ChordGene]):
        # Rewards note of the chords on being on scale
        max_fit = 0
        for scale in scales:
            fit = sum([
                sum([pitch in scale for pitch in gene.pitches])
                for gene in genes
            ])
            max_fit = max(max_fit, fit)
        return max_fit * 2

    def fit_to_scale_chords(genes: List[ChordGene]):
        # Rewards the chord for being one of the fitted in scale chords
        max_fit = 0
        for scale, root in scales_with_roots:
            fit = sum([
                frozenset(gene.pitches) in scale.get_chords(root)
                for gene in genes
            ])
            max_fit = max(max_fit, fit)
        return max_fit * 10

    dissonance = frozenset({1, 2, 6, 10, 11})

    def fit_to_current_notes(genes: List[ChordGene]):
        # Punishes dissonance intervals and fits chords to notes in bar
        s = 0
        for i, gene in enumerate(genes):
            current_notes = bar_to_notes[i]
            for n1, n2 in combinations(chain(current_notes, gene.pitches), 2):
                if abs(n1.value - n2.value) in dissonance:
                    s -= 5
                else:
                    s += 1
            for chord_note in gene.pitches:
                s += int(chord_note in current_notes) * 2
        return s * 5

    def punish_same_notes(genes: List[ChordGene]):
        fit = sum([-1 if len(set(gene.pitches)) != 3 else 0 for gene in genes])
        return fit * 50

    def num_of_unique_chords(genes: List[ChordGene]):
        # Number of unique chords should be between MIN and MAX
        num = len(set([frozenset(gene.pitches) for gene in genes]))
        ans = 0
        MIN, MAX = 4, 6
        if ans in range(MIN, MAX + 1):
            return 0
        ans -= max(num - MAX, 0)
        ans -= max(MIN - num, 0)
        return ans * 50

    def same_chord(genes: List[ChordGene]):
        # Rewards same chord playing again
        last_pitches = genes[0].pitches
        s = 0
        for gene in genes[1:]:
            if last_pitches == gene.pitches:
                s += 1
        return s * 30

    funcs = [
        fit_to_scale,
        punish_same_notes,
        fit_to_current_notes,
        num_of_unique_chords,
        fit_to_scale_chords,
        same_chord
    ]

    def fitness(chromo: Chromosome) -> float:
        genes: List[ChordGene] = chromo.genes
        return sum([f(genes) for f in funcs])

    def print_fitness(chromo: Chromosome):
        genes: List[ChordGene] = chromo.genes
        for f in funcs:
            print(f.__name__.replace("_", " "), ":", f(genes))

    return fitness, print_fitness


def chromosome_to_chords(genes: List[ChordGene]) -> List[List[Note]]:
    notes = []
    for chord_gene in genes:
        chord = []
        for pitch in chord_gene.pitches:
            chord.append(Note(Pitch(pitch.value + 12 * 5)))
        notes.append(chord)
    return notes


def main():
    FILE_INPUT = 'input3.mid'
    FILE_OUTPUT = 'output.mid'

    mid = MidiFile(FILE_INPUT, clip=True)
    mid.type = 1
    melody_track = mid.tracks[1]

    # Get notes from midi
    notes, velocity, duration = Midi.track_to_notes(melody_track, mid.ticks_per_beat)
    print("\n".join([str(note) for note in notes]))

    # Find possible scales
    scales, scales_with_root = possible_scales(notes, [major_scale, minor_scale])
    print("Fitted scales:")
    print(*[" ".join([str(pitch.name) for pitch in scale]) for scale in scales], sep="\n")

    print()
    # Get fitness function
    fitness, print_fitness = chord_fit(notes)
    print()

    # Take one of the fitted scales
    scale, root = scales_with_root[0]
    # Get all suitable chords for that scale
    chords = list(scale.get_chords(root))
    ChordGene.possible_chords = chords

    ga = GA(fitness, gene_generate=ChordGene.generate)
    # Run genetic algorithm
    print("Running genetic algorithm...")
    best = ga.run(20, int(duration), iters=2_000)
    best_chromosome = best["chromosome"]
    print_fitness(best_chromosome)

    # Convert chromosome to chords
    chords = chromosome_to_chords(best_chromosome.genes)
    print("Best fitness:", best["fitness"])
    print("Final chords:")
    print(*[" ".join([str(note.pitch.name) for note in chord]) for chord in chords], sep="\n")

    mid.type = 1
    # Convert chords to midi
    melody_track = Midi.chords_to_track(chords, mid.ticks_per_beat, velocity=int(velocity * 0.8))
    mid.tracks.append(melody_track)

    mid.save(FILE_OUTPUT)


if __name__ == '__main__':
    main()
