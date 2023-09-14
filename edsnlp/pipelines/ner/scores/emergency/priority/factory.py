import re
from typing import Any, Callable, List, Optional, Union

from edsnlp.core import PipelineProtocol, registry
from edsnlp.pipelines.base import SpanSetterArg
from edsnlp.pipelines.ner.scores.base_score import SimpleScoreMatcher

from .patterns import regex, score_normalization_str, value_extract

DEFAULT_CONFIG = dict(
    regex=regex,
    value_extract=value_extract,
    score_normalization=score_normalization_str,
    attr="NORM",
    window=7,
    ignore_excluded=False,
    ignore_space_tokens=False,
    flags=0,
    label="emergency_priority",
    span_setter={"ents": True, "emergency_priority": True},
)


@registry.factory.register(
    "eds.emergency_priority",
    assigns=["doc.ents", "doc.spans"],
    deprecated=[
        "emergency.priority",
        "eds.emergency.priority",
    ],
)
def create_component(
    nlp: PipelineProtocol,
    name: Optional[str] = None,
    *,
    regex: List[str] = regex,
    value_extract: str = value_extract,
    score_normalization: Union[
        str, Callable[[Union[str, None]], Any]
    ] = score_normalization_str,
    attr: str = "NORM",
    window: int = 7,
    ignore_excluded: bool = False,
    ignore_space_tokens: bool = False,
    flags: Union[re.RegexFlag, int] = 0,
    label: str = "emergency_priority",
    span_setter: SpanSetterArg = {"ents": True, "emergency_priority": True},
):
    """
    Matcher for explicit mentions of the French priority emergency score.

    Examples
    --------

    ```python
    import spacy

    nlp = spacy.blank("eds")
    nlp.add_pipe("eds.emergency_priority")
    ```

    Parameters
    ----------
    nlp : PipelineProtocol
        The pipeline object
    name : str
        The name of the component
    regex : List[str]
        A list of regexes to identify the score
    attr : str
        Whether to match on the text ('TEXT') or on the normalized text ('NORM')
    value_extract : str
        Regex with capturing group to get the score value
    score_normalization : Union[str, Callable[[Union[str,None]], Any]]
        Function that takes the "raw" value extracted from the `value_extract`
        regex and should return:

        - None if no score could be extracted
        - The desired score value else
    window : int
        Number of token to include after the score's mention to find the
        score's value
    ignore_excluded : bool
        Whether to ignore excluded spans when matching
    ignore_space_tokens : bool
        Whether to ignore space tokens when matching
    flags : Union[re.RegexFlag, int]
        Regex flags to use when matching
    label : str
        Label name to use for the `Span` object and the extension
    span_setter: SpanSetterArg
        How to set matches on the doc

    Returns
    -------
    SimpleScoreMatcher
    """
    return SimpleScoreMatcher(
        nlp,
        name=name,
        regex=regex,
        value_extract=value_extract,
        score_normalization=score_normalization,
        attr=attr,
        window=window,
        ignore_excluded=ignore_excluded,
        ignore_space_tokens=ignore_space_tokens,
        flags=flags,
        label=label,
        span_setter=span_setter,
    )
