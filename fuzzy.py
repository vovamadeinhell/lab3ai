"""Нечеткий вывод без сторонних библиотек. Python 3.10+."""
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Callable


IMPLICATIONS: dict[str, Callable[[float, float], float]] = {
    "mamdani": min,
    "larsen": lambda a, b: a * b,
    "kleene_dienes": lambda a, b: max(1 - a, b),
    "lukasiewicz": lambda a, b: min(1.0, 1 - a + b),
}
METHODS = ("max_min", "max_product", "activation")
AGGREGATIONS = ("max", "min", "probor", "bounded_sum")
AND_OPERATORS = ("min", "product")


def finite(value, label="Число") -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}: требуется конечное число")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{label}: требуется конечное число")
    return value


def aggregate(values, operator: str) -> float:
    values = tuple(values)
    if not values:
        raise ValueError("Нельзя агрегировать пустой список")
    if operator == "max":
        return max(values)
    if operator == "min":
        return min(values)
    if operator == "probor":
        return 1 - math.prod(1 - v for v in values)
    if operator == "bounded_sum":
        return min(1.0, sum(values))
    raise ValueError(f"Неизвестная агрегация: {operator}")


@dataclass(frozen=True)
class Membership:
    kind: str
    points: tuple[float, ...]

    def __post_init__(self):
        count = {"triangle": 3, "trapezoid": 4}.get(self.kind)
        if count is None or len(self.points) != count:
            raise ValueError("Множество: triangle с 3 или trapezoid с 4 параметрами")
        points = tuple(finite(x, "Параметр множества") for x in self.points)
        if list(points) != sorted(points) or points[0] == points[-1]:
            raise ValueError("Параметры должны возрастать; ширина должна быть положительной")
        object.__setattr__(self, "points", points)

    def __call__(self, x: float) -> float:
        if self.kind == "triangle":
            a, b, d = self.points
            c = b
        else:
            a, b, c, d = self.points
        # Совпавшие a=b и c=d задают плечи, включая крайние точки.
        if x < a or x > d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if x < b:
            return (x - a) / (b - a)
        return (d - x) / (d - c)


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    unit: str
    domain: tuple[float, float]
    step: float
    terms: dict[str, Membership]

    def grid(self) -> list[float]:
        lo, hi = self.domain
        count = int(math.ceil((hi - lo) / self.step))
        # Включаем все вершины функций принадлежности независимо от шага.
        return sorted({lo, hi, *(min(hi, lo + i * self.step)
                                for i in range(count + 1)),
                       *(x for mf in self.terms.values() for x in mf.points
                         if lo <= x <= hi)})


@dataclass(frozen=True)
class Rule:
    conditions: tuple[tuple[str, str], ...]
    conclusion: str


@dataclass(frozen=True)
class System:
    title: str
    inputs: dict[str, Variable]
    output: Variable
    rules: tuple[Rule, ...]
    example: dict[str, float]


def load_system(path: Path | str) -> System:
    """Вся предметная база (множества и правила) читается из UTF-8 JSON-текста."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Повторяющийся ключ: {key}")
            result[key] = value
        return result

    def variable(name, data):
        lo, hi = (finite(x, "Граница диапазона") for x in data["range"])
        step = finite(data["step"], "Шаг")
        if lo >= hi or step <= 0 or (hi - lo) / step > 10000:
            raise ValueError("Неверный диапазон или шаг; допускается до 10000 интервалов")
        terms = {name: Membership(mf["type"], tuple(mf["params"]))
                 for name, mf in data["terms"].items()}
        if not terms:
            raise ValueError("У переменной нет нечетких множеств")
        if any(m.points[0] < lo or m.points[-1] > hi for m in terms.values()):
            raise ValueError("Множество выходит за диапазон переменной")
        return Variable(name, data.get("label", name), data.get("unit", ""),
                        (lo, hi), step, terms)

    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"),
                          object_pairs_hook=unique_keys)
        inputs = {name: variable(name, v) for name, v in data["inputs"].items()}
        output = variable(data["output"]["name"], data["output"])
        if not inputs or output.name in inputs:
            raise ValueError("Нужны входы и отдельная выходная переменная")
        rules = []
        for row in data["rules"]:
            conditions = tuple(tuple(c) for c in row["if"])
            if not conditions or len({c[0] for c in conditions}) != len(conditions):
                raise ValueError("Пустое условие или повтор входа в правиле")
            for name, term in conditions:
                if name not in inputs or term not in inputs[name].terms:
                    raise ValueError(f"Неизвестное условие: {name}={term}")
            if row["then"] not in output.terms:
                raise ValueError(f"Неизвестное заключение: {row['then']}")
            rules.append(Rule(conditions, row["then"]))
        if not rules:
            raise ValueError("База правил пуста")
        example = {k: finite(v) for k, v in data["example"].items()}
        if set(example) != set(inputs):
            raise ValueError("Пример должен задавать все входные переменные")
        for name, value in example.items():
            if not inputs[name].domain[0] <= value <= inputs[name].domain[1]:
                raise ValueError("Пример выходит за диапазон")
        return System(data["title"], inputs, output, tuple(rules), example)
    except (KeyError, TypeError, ValueError, IndexError) as error:
        raise ValueError(f"Ошибка базы {path}: {error}") from error


def centroid(xs, membership) -> tuple[float | None, float]:
    """Точные площадь и момент полилинии, интерполирующей дискретное множество."""
    if len(xs) != len(membership) or len(xs) < 2:
        raise ValueError("Несогласованные размеры сетки и множества")
    area = moment = 0.0
    for x0, x1, a, b in zip(xs, xs[1:], membership, membership[1:]):
        if x1 <= x0:
            raise ValueError("Сетка должна строго возрастать")
        dx = x1 - x0
        area += dx * (a + b) / 2
        moment += dx * ((2 * x0 + x1) * a + (x0 + 2 * x1) * b) / 6
    return (moment / area if area > 1e-12 else None), area


@dataclass
class Result:
    xs: list[float]
    membership: list[float]
    crisp: float | None
    area: float
    degrees: dict[str, dict[str, float]]
    alphas: list[float]
    warnings: list[str]


def infer(system: System, values: dict[str, float], *, method="activation",
          implication="mamdani", aggregation="max", and_operator="min",
          fuzzy_inputs: dict[str, Membership] | None = None) -> Result:
    """FATI-композиция общего отношения или вывод по уровням истинности.

    В композиции R(x,y)=Agg_i I(A_i(x),B_i(y)); затем sup_x T(A'(x),R(x,y)).
    При четком входе A' — синглтон точно в заданной точке, даже вне узлов сетки.
    Для многих входов композиция не строится (экспоненциальная сетка);
    используется предусмотренный заданием метод activation.
    """
    if method not in METHODS or implication not in IMPLICATIONS:
        raise ValueError("Неизвестный механизм или импликация")
    if aggregation not in AGGREGATIONS or and_operator not in AND_OPERATORS:
        raise ValueError("Неизвестный оператор")
    if method != "activation" and len(system.inputs) != 1:
        raise ValueError("Для нескольких входов выберите activation")
    fuzzy_inputs = fuzzy_inputs or {}
    if set(values) != set(system.inputs) or not set(fuzzy_inputs) <= set(values):
        raise ValueError("Нужно задать ровно все входы системы")
    for name, value in values.items():
        value = finite(value, name)
        lo, hi = system.inputs[name].domain
        if not lo <= value <= hi:
            raise ValueError(f"{name}: допустимый диапазон [{lo}, {hi}]")
    for name, mf in fuzzy_inputs.items():
        lo, hi = system.inputs[name].domain
        if mf.points[0] < lo or mf.points[-1] > hi:
            raise ValueError(f"Нечеткое наблюдение {name} выходит за диапазон")

    samples = {}
    degrees = {}
    for name, var in system.inputs.items():
        if name in fuzzy_inputs:
            observation = fuzzy_inputs[name]
            nodes = set(var.grid()) | set(observation.points) | {values[name]}
            # Добавляем пересечения A' и термов: sup min(A',A_i) тогда
            # вычисляется точно для кусочно-линейных функций, без потери пика
            # совместимости между узлами равномерной сетки.
            for mf in var.terms.values():
                knots = sorted(set(observation.points) | set(mf.points))
                for left, right in zip(knots, knots[1:]):
                    dl = observation(left) - mf(left)
                    dr = observation(right) - mf(right)
                    if dl * dr < 0:
                        nodes.add(left + (right - left) * dl / (dl - dr))
            grid = sorted(nodes)
            samples[name] = [(x, observation(x)) for x in grid if observation(x) > 0]
        else:
            samples[name] = [(values[name], 1.0)]
        if not samples[name]:
            raise ValueError("Наблюдение не пересекается с сеткой; уменьшите шаг")
        degrees[name] = {
            term: max(min(mu, mf(x)) for x, mu in samples[name])
            for term, mf in var.terms.items()
        }
    and_fn = min if and_operator == "min" else math.prod
    alphas = [and_fn(degrees[n][t] for n, t in rule.conditions)
              for rule in system.rules]
    xs = system.output.grid()
    consequents = [[system.output.terms[r.conclusion](y) for y in xs]
                   for r in system.rules]
    impl = IMPLICATIONS[implication]
    if method == "activation":
        # Нулевые alpha НЕ пропускаются: I(0,b)=1 у логических импликаций.
        membership = [aggregate((impl(a, b[j]) for a, b in zip(alphas, consequents)),
                                aggregation) for j in range(len(xs))]
    else:
        name = next(iter(system.inputs))
        membership = [0.0] * len(xs)
        for x, observation_mu in samples[name]:
            antecedents = [and_fn(system.inputs[n].terms[t](x)
                                 for n, t in rule.conditions) for rule in system.rules]
            for j in range(len(xs)):
                relation = aggregate((impl(a, b[j])
                                      for a, b in zip(antecedents, consequents)), aggregation)
                composed = (min(observation_mu, relation) if method == "max_min"
                            else observation_mu * relation)
                membership[j] = max(membership[j], composed)
    crisp, area = centroid(xs, membership)
    warnings = []
    if crisp is None:
        warnings.append("Выходное множество пусто; центр тяжести не определён.")
    elif max(membership) - min(membership) < 1e-10:
        warnings.append("Выходное множество постоянно: результат не различает рекомендации.")
    if implication in ("kleene_dienes", "lukasiewicz") and aggregation != "min":
        warnings.append("Для логических импликаций I(0,b)=1. Дизъюнктивная агрегация "
                        "может дать постоянный выход; сравните агрегацию min.")
    return Result(xs, membership, crisp, area, degrees, alphas, warnings)


def describe_rules(system: System) -> list[str]:
    return [f"{i}. ЕСЛИ " + " И ".join(f"{n}={t}" for n, t in r.conditions)
            + f" ТО {system.output.name}={r.conclusion}"
            for i, r in enumerate(system.rules, 1)]
