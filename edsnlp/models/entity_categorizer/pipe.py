from itertools import islice
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy
from spacy import Language
from spacy.pipeline.trainable_pipe import TrainablePipe
from spacy.scorer import PRFScore
from spacy.tokens.doc import Doc
from spacy.training.example import Example
from spacy.vocab import Vocab
from thinc.api import Model, Optimizer
from thinc.model import set_dropout_rate
from thinc.types import Floats2d, Ints1d
from wasabi import Printer

Doc.set_extension("rel", default={}, force=True)
msg = Printer()


@Language.factory(
    "entity_categorizer",
    requires=["doc.ents", "token.ent_iob", "token.ent_type"],
    assigns=["doc._.rel"],
    default_score_weights={
        "rel_micro_p": None,
        "rel_micro_r": None,
        "rel_micro_f": None,
    },
)
def make_entity_categorizer(
    nlp: Language, name: str, model: Model, *, threshold: float
):
    """Construct a EntityCategorizer component."""
    return EntityCategorizer(nlp.vocab, model, name, threshold=threshold)


# noinspection PyMethodOverriding,PyProtectedMember
class EntityCategorizer(TrainablePipe):
    def __init__(
        self,
        vocab: Vocab,
        model: Model,
        name: str = "rel",
        *,
        threshold: float,
    ) -> None:
        """Initialize an entity categorizer."""
        self.vocab = vocab
        self.model = model
        self.name = name
        self.cfg = {
            "attrs": [],
            "threshold": threshold,
            "filter_spans_labels": [],
            "filter_spans_ents": [],
        }

    @property
    def filter_spans_labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["filter_spans_labels"])

    @property
    def filter_ents_labels(self) -> Tuple[str]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["filter_ents_labels"])

    @property
    def attrs(self) -> Tuple[Tuple[str, str]]:
        """Returns the labels currently added to the component."""
        return tuple(self.cfg["attrs"])

    @property
    def threshold(self) -> float:
        """Returns the threshold above which a prediction is seen as 'True'."""
        return self.cfg["threshold"]

    def add_attr(self, attr_key: str, attr_value) -> int:
        """Add a new label to the pipe."""
        if not isinstance(attr_key, str):
            raise ValueError(
                "Only strings can be added as attribute keys to the EntityCategorizer"
            )

        if not isinstance(attr_value, str):
            raise ValueError(
                "Only strings can be added as attribute values to the EntityCategorizer"
            )

        if attr_key in self.attrs:
            return 0
        self.cfg["attrs"] = list(self.attrs) + [(attr_key, attr_value)]
        return 1

    def __call__(self, doc: Doc) -> Doc:
        predictions = self.predict([doc])
        self.set_annotations([doc], predictions)
        return doc

    def get_instances(self, docs: Iterable[Doc]):
        return [
            list(set(
                [
                    span
                    for span in doc.ents
                    if span.label in self.filter_ents_labels]
                + [
                    span
                    for label in self.filter_spans_labels
                    for span in doc.spans[label]
                ]
            ))
            for doc in docs
        ]

    def predict(self, docs: Iterable[Doc]) -> (Ints1d, Ints1d, Floats2d):
        """Apply the pipeline's model to a batch of docs, without modifying them."""
        # DONE

        scores = self.model.predict(docs, self.get_instances(docs))
        return self.model.ops.asarray(scores)

    def set_annotations(
        self, docs: Iterable[Doc], scores: (Ints1d, Ints1d, Ints1d, Floats2d)
    ) -> None:
        """Modify a batch of `Doc` objects, using pre-computed scores."""
        # DONE
        c = 0
        docs = list(docs)
        doc_indices, begins, ends, scores = scores
        for (attr_key, attr_value), label_scores in zip(self.attrs, scores):
            keep = scores > 0
            for doc_idx, b, e, label_score in zip(
                doc_indices[keep], begins[keep], ends[keep], scores[keep]
            ):
                docs[doc_idx][b:e]._.set(attr_key, attr_value)

    def update(
        self,
        examples: Iterable[Example],
        *,
        drop: float = 0.0,
        set_annotations: bool = False,
        sgd: Optional[Optimizer] = None,
        losses: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Learn from a batch of documents and gold-standard information,
        updating the pipe's model. Delegates to predict and get_loss."""
        if losses is None:
            losses = {}
        losses.setdefault(self.name, 0.0)
        set_dropout_rate(self.model, drop)

        # run the model
        docs = [eg.predicted for eg in examples]
        predictions, backprop = self.model.begin_update(docs)
        loss, gradient = self.get_loss(examples, predictions)
        backprop(gradient)
        if sgd is not None:
            self.model.finish_update(sgd)
        losses[self.name] += loss
        if set_annotations:
            self.set_annotations(docs, predictions)
        return losses

    def get_loss(self, examples: Iterable[Example], scores) -> Tuple[float, float]:
        """Find the loss and gradient of loss for the batch of documents and
        their predicted scores."""
        truths = self._examples_to_truth(examples)
        gradient = scores - truths
        mean_square_error = (gradient**2).sum(axis=1).mean()
        return float(mean_square_error), gradient

    def initialize(
        self,
        get_examples: Callable[[], Iterable[Example]],
        *,
        nlp: Language = None,
        labels: Optional[List[str]] = None,
    ):
        """Initialize the pipe for training, using a representative set
        of data examples.
        """
        if labels is not None:
            for label in labels:
                self.add_label(label)
        else:
            for example in get_examples():
                categories = example.reference._.rel
                for indices, label_dict in categories.items():
                    for label in label_dict.keys():
                        self.add_label(label)
        self._require_labels()

        subbatch = list(islice(get_examples(), 10))
        doc_sample = [eg.reference for eg in subbatch]
        label_sample = self._examples_to_truth(subbatch)
        if label_sample is None:
            raise ValueError(
                "Call begin_training with relevant entities and categories annotated in "
                "at least a few reference examples!"
            )
        self.model.initialize(X=doc_sample, Y=label_sample)

    def _examples_to_truth(self, examples: List[Example]) -> Optional[numpy.ndarray]:
        # check that there are actually any candidate instances in this batch of examples
        nr_instances = 0
        for eg in examples:
            nr_instances += len(self.model.attrs["get_instances"](eg.reference))
        if nr_instances == 0:
            return None

        truths = numpy.zeros((nr_instances, len(self.labels)), dtype="f")
        c = 0
        for i, eg in enumerate(examples):
            for (e1, e2) in self.model.attrs["get_instances"](eg.reference):
                gold_label_dict = eg.reference._.rel.get((e1.start, e2.start), {})
                for j, label in enumerate(self.labels):
                    truths[c, j] = gold_label_dict.get(label, 0)
                c += 1

        truths = self.model.ops.asarray(truths)
        return truths

    def score(self, examples: Iterable[Example], **kwargs) -> Dict[str, Any]:
        """Score a batch of examples."""
        return score_categories(examples, self.threshold)


def score_categories(examples: Iterable[Example], threshold: float) -> Dict[str, Any]:
    """Score a batch of examples."""
    micro_prf = PRFScore()
    for example in examples:
        gold = example.reference._.rel
        pred = example.predicted._.rel
        for key, pred_dict in pred.items():
            gold_labels = [k for (k, v) in gold.get(key, {}).items() if v == 1.0]
            for k, v in pred_dict.items():
                if v >= threshold:
                    if k in gold_labels:
                        micro_prf.tp += 1
                    else:
                        micro_prf.fp += 1
                else:
                    if k in gold_labels:
                        micro_prf.fn += 1
    return {
        "rel_micro_p": micro_prf.precision,
        "rel_micro_r": micro_prf.recall,
        "rel_micro_f": micro_prf.fscore,
    }
