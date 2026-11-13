"""Меню лабораторной №3 и воспроизводимый пакет экспериментов (--demo)."""
import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path

from fuzzy import (AGGREGATIONS, AND_OPERATORS, IMPLICATIONS, METHODS, Membership,
                   describe_rules, infer, load_system)


BASE = Path(__file__).resolve().parent
BASES = {"single": BASE / "single_input.txt", "multi": BASE / "multi_input.txt"}
METHOD_LABELS = {"max_min": "Композиционный max-min",
                 "max_product": "Композиционный max-product",
                 "activation": "Уровни истинности предпосылок"}


def print_result(system, values, result, method, implication, aggregation, and_operator="min"):
    print(f"\n{system.title}")
    print(f"Входы: {values}")
    print(f"Механизм: {METHOD_LABELS[method]}; импликация: {implication}; "
          f"агрегация: {aggregation}; И: {and_operator}")
    print("Степени принадлежности (для нечеткого входа — степени совместимости):")
    for name, degrees in result.degrees.items():
        print(f"  {name}: " + "; ".join(f"{t}={v:.6f}" for t, v in degrees.items()))
    print("Уровни истинности всех правил:")
    for i, (rule, alpha) in enumerate(zip(system.rules, result.alphas), 1):
        print(f"  R{i:02}: alpha={alpha:.6f} -> {rule.conclusion}")
    print(f"Площадь выходного множества: {result.area:.6f}")
    print("Центр тяжести: " + (f"{result.crisp:.6f} {system.output.unit}"
                                   if result.crisp is not None else "не определён"))
    for warning in result.warnings:
        print(f"Примечание: {warning}")


def experiments():
    """Одинаковые входы внутри каждой серии; меняется только указанный оператор."""
    single = load_system(BASES["single"])
    multi = load_system(BASES["multi"])
    data = json.loads(BASES["single"].read_text(encoding="utf-8"))
    fuzzy = {n: Membership(v["type"], tuple(v["params"]))
             for n, v in data["fuzzy_example"].items()}
    rows = []

    def add(case, system, method, implication="mamdani", aggregation="max",
            and_operator="min", observation=None):
        result = infer(system, system.example, method=method, implication=implication,
                       aggregation=aggregation, and_operator=and_operator,
                       fuzzy_inputs=observation)
        rows.append({"case": case, "method": method, "implication": implication,
                     "aggregation": aggregation, "and_operator": and_operator,
                     "inputs": system.example,
                     "fuzzy_inputs": {n: asdict(mf) for n, mf in (observation or {}).items()},
                     "result": asdict(result)})

    for case, observation in (("single_crisp", None), ("single_fuzzy", fuzzy)):
        for method in METHODS:
            add(case, single, method, observation=observation)
    for implication in IMPLICATIONS:
        for aggregation in AGGREGATIONS:
            add("single_implications", single, "activation", implication, aggregation)
            add("multi_implications", multi, "activation", implication, aggregation)
    for method in ("max_min", "max_product"):
        for implication in IMPLICATIONS:
            for aggregation in ("max", "min"):
                add("composition_implications", single, method, implication,
                    aggregation, observation=fuzzy)
    add("multi_product_and", multi, "activation", and_operator="product")
    return rows


def save_experiments(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fields = ["case", "method", "implication", "aggregation", "and_operator", "centroid", "area"]
    with (output_dir / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in fields[:5]},
                             "centroid": row["result"]["crisp"], "area": row["result"]["area"]})
    with (output_dir / "output_sets.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "case", "y", "membership"])
        for i, row in enumerate(rows, 1):
            for y, mu in zip(row["result"]["xs"], row["result"]["membership"]):
                writer.writerow([i, row["case"], y, mu])


def demo(output_dir):
    rows = experiments()
    save_experiments(rows, output_dir)
    print("Воспроизводимые эксперименты лабораторной №3")
    print("Одномерный вход: 77 °C; нечеткое наблюдение: triangle(71,77,83).")
    print("Три входа: температура 77 °C; загрузка 68 %; запылённость 6.5 балла.")
    print("case | method | implication | aggregation | И | centroid | area")
    for row in rows:
        result = row["result"]
        crisp = "не определён" if result["crisp"] is None else f"{result['crisp']:.6f}"
        print(" | ".join(str(row[k]) for k in
                         ("case", "method", "implication", "aggregation", "and_operator"))
              + f" | {crisp} | {result['area']:.6f}")
    print("\nПодробный ход базового вывода для трёх входов:")
    system = load_system(BASES["multi"])
    result = infer(system, system.example)
    print_result(system, system.example, result, "activation", "mamdani", "max")
    print(f"\nСохранено экспериментов: {len(rows)}. Файлы: {output_dir}")


def select(label, options):
    print(label)
    for i, value in enumerate(options, 1):
        print(f"  {i}. {value}")
    while True:
        raw = input("Номер (Enter — 1): ").strip() or "1"
        if raw.isdecimal() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        print("Введите номер из списка.")


def number(label, lo, hi, default):
    while True:
        raw = input(f"{label} [{lo:g}..{hi:g}] (Enter — {default:g}): ").strip()
        try:
            value = float(raw.replace(",", ".")) if raw else default
            if lo <= value <= hi:
                return value
        except ValueError:
            pass
        print("Введите конечное число из допустимого диапазона.")


def consultation(kind):
    system = load_system(BASES[kind])
    values = {name: number(f"{var.label}, {var.unit}", *var.domain, system.example[name])
              for name, var in system.inputs.items()}
    observation = {}
    if kind == "single":
        center = values["temperature"]
        lo, hi = system.inputs["temperature"].domain
        width = number("Полуширина нечеткого наблюдения, °C (0 — точное значение)",
                       0, min(center - lo, hi - center), 0)
        if width > 0:
            observation["temperature"] = Membership("triangle", (center-width, center, center+width))
    method = select("Механизм вывода", METHODS) if kind == "single" else "activation"
    implication = select("Импликация", tuple(IMPLICATIONS))
    aggregation = select("Агрегация выходов правил", AGGREGATIONS)
    and_operator = select("Оператор И", AND_OPERATORS) if kind == "multi" else "min"
    result = infer(system, values, method=method, implication=implication,
                   aggregation=aggregation, and_operator=and_operator,
                   fuzzy_inputs=observation)
    print_result(system, values, result, method, implication, aggregation, and_operator)
    if observation:
        print("Нечеткое наблюдение: " + str({k: v.points for k, v in observation.items()}))


def menu():
    while True:
        print("\nЛАБОРАТОРНАЯ №3: НЕЧЕТКИЕ СИСТЕМЫ ЛОГИЧЕСКОГО ВЫВОДА\n"
              "1. Одна входная переменная\n2. Три входные переменные\n"
              "3. Сравнение механизмов и импликаций с сохранением результатов\n"
              "4. Показать нечеткие множества и правила\n0. Выход")
        choice = input("Выберите действие: ").strip()
        try:
            if choice == "0":
                return
            if choice in ("1", "2"):
                consultation("single" if choice == "1" else "multi")
            elif choice == "3":
                demo(BASE / "results")
            elif choice == "4":
                kind = select("Модель", tuple(BASES))
                system = load_system(BASES[kind])
                for var in [*system.inputs.values(), system.output]:
                    print(f"\n{var.name}: {var.label}, {var.domain} {var.unit}, шаг {var.step}")
                    for term, mf in var.terms.items():
                        print(f"  {term}: {mf.kind}{mf.points}")
                print("\n".join(describe_rules(system)))
            else:
                print("Неизвестный пункт меню.")
        except (ValueError, OSError) as error:
            print(f"Ошибка: {error}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="Выполнить все сравнения и сохранить CSV/JSON")
    parser.add_argument("--output-dir", type=Path, default=BASE / "results")
    args = parser.parse_args()
    try:
        if args.demo:
            demo(args.output_dir)
        else:
            menu()
    except (EOFError, KeyboardInterrupt):
        print("\nРабота завершена.")
    except (ValueError, OSError) as error:
        parser.exit(1, f"Ошибка: {error}\n")


if __name__ == "__main__":
    main()
