from __future__ import annotations

# =============================================================================
# Fichier: solver.py
# Objet: Implémentation d'un heuristique de préparation de commandes (batching
#        et picking) suivant la logique décrite par l'utilisateur.
#
# Idée générale (résumé en français):
# 1) Pour chaque tournée (tant que toutes les commandes ne sont pas terminées),
#    à l'arrêt, on « sélectionne » des commandes à activer en fonction des
#    produits les plus proches du point de départ. On crée jusqu'à K colis (K
#    étant le nombre de cartons par chariot). Ici, on crée initialement un
#    carton par commande active (pas de mélange de commandes dans un même
#    carton) et on pourra ouvrir de nouveaux cartons pour une commande si
#    nécessaire pendant le picking, tant que K n'est pas dépassé.
# 2) En mouvement, on visite itérativement le produit le plus proche parmi ceux
#    dont les commandes actives ont encore besoin. Arrivé sur ce produit, on
#    remplit les colis des commandes concernées en respectant les capacités
#    (poids/volume) de chaque carton. On ouvre de nouveaux cartons si besoin et
#    si on n'a pas encore atteint K.
# 3) Quand plus aucun progrès n'est possible (plus de demande ou plus de place),
#    on termine la tournée, on ajoute les cartons remplis et on recommence une
#    nouvelle tournée si des commandes restent incomplètes.
# 4) La sortie est formatée via transform_output.Output afin de générer un
#    fichier solution compatible avec le checker fourni.
#
# Remarques:
# - On ne mélange pas les commandes dans un même carton dans cette version (on
#   pourrait l'étendre si mixed_orders_allowed est vrai).
# - Les métriques travelled_distance et crossed_locations ne sont pas
#   finement calculées ici (fixées à 0). On pourrait les calculer précisément
#   si nécessaire en retraçant l'itinéraire suivi.
# =============================================================================

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set, Any

from read_instance import Instance
from transform_output import Output, Tour, Boxe, BoxeProduct, Order as OutOrder


# --------- Internal data models (lightweight) ---------

@dataclass
class Demand:
    # Remaining qty for a product within an order
    product_id: int
    quantity: int


@dataclass
class OrderState:
    id: int
    # product_id -> remaining quantity
    remaining: Dict[int, int]

    def is_done(self) -> bool:
        """Retourne True si l'ensemble des quantités restantes de la commande
        est nul (toutes les demandes sont satisfaites)."""
        return all(qty <= 0 for qty in self.remaining.values())


@dataclass
class BoxState:
    id: int
    order_id: Optional[int]  # None if not assigned (unused: we keep per-order boxes)
    weight: int = 0
    volume: int = 0
    products: Dict[int, int] = field(default_factory=dict)  # product_id -> qty in box

    def add(self, prod_w: int, prod_v: int, prod_id: int, qty: int) -> None:
        """Ajoute qty unités d'un produit dans ce carton et met à jour le
        poids et volume utilisés."""
        self.weight += prod_w * qty
        self.volume += prod_v * qty
        self.products[prod_id] = self.products.get(prod_id, 0) + qty


# --------- Résolveur de plus courts chemins ---------

class ShortestPathResolver:
    """Résout des distances minimales entre lieux via la matrice fournie.
    Hypothèse: la matrice des plus courts chemins est complète pour toutes paires.
    Retourne None si la paire est absente (cas inattendu dans ce contexte).
    """

    def __init__(self, instance: Instance) -> None:
        self.known: Dict[Tuple[int, int], int] = {}
        for p in instance.graph["shortest_distances"]:
            a = p["idDeparture"]
            b = p["idArrival"]
            w = p["distance"]
            self.known[(a, b)] = w
            self.known[(b, a)] = w

    def dist(self, a: int, b: int) -> Optional[int]:
        if a == b:
            return 0
        return self.known.get((a, b))


def products_by_id(instance: Instance) -> Dict[int, dict]:
    """Indexe les produits par leur identifiant: {product_id: produit}."""
    return {p["id"]: p for p in instance.products}


def products_sorted_from_start(instance: Instance, resolver: ShortestPathResolver) -> List[int]:
    """Retourne la liste des IDs produit triée par proximité au dépôt de départ.
    Sert pour la phase d'affectation à l'arrêt (prioriser les produits proches)."""
    start = instance.graph["departing_depot"]
    prods = products_by_id(instance)
    def key_fun(pid: int) -> float:
        d = resolver.dist(start, prods[pid]["id_loc"])  # Optional[int]
        return float(d) if d is not None else float("inf")
    return sorted(prods.keys(), key=key_fun)


def build_orders_state(instance: Instance) -> Dict[int, OrderState]:
    """Construit l'état initial des commandes: pour chaque commande, un
    dictionnaire {product_id -> quantité restante}."""
    states: Dict[int, OrderState] = {}
    for o in instance.orders:
        rem: Dict[int, int] = {}
        for p in o["products"]:
            pid = p["product"]["id"]
            qty = p["quantity"]
            rem[pid] = rem.get(pid, 0) + qty
        states[o["id"]] = OrderState(id=o["id"], remaining=rem)
    return states


def order_contains_product(order_state: OrderState, product_id: int) -> bool:
    """Indique si la commande a encore besoin du produit donné."""
    return order_state.remaining.get(product_id, 0) > 0


def needed_products(order_state: OrderState) -> List[int]:
    """Liste des produits dont la quantité restante est positive."""
    return [pid for pid, qty in order_state.remaining.items() if qty > 0]


def next_nearest_product(current_loc: int, candidate_products: Set[int], prods_by_id: Dict[int, dict], resolver: ShortestPathResolver) -> Optional[int]:
    """Parmi un ensemble de produits candidats, retourne l'ID du plus proche
    depuis la position courante."""
    if not candidate_products:
        return None
    best_pid: Optional[int] = None
    best_d: Optional[int] = None
    for pid in candidate_products:
        loc = prods_by_id[pid]["id_loc"]
        d = resolver.dist(current_loc, loc)
        if d is None:
            continue
        if best_d is None or d < best_d:
            best_d = d
            best_pid = pid
    return best_pid


# --------- Core heuristic (as per the user's description) ---------

def solve_instance(filename: str) -> Output:
    """Résout une instance en appliquant l'heuristique décrite.

    Étapes principales:
    - Pré-calculer la distance entre lieux, l'index des produits, et la
      liste des produits triés par proximité au départ.
    - Tant que des commandes restent à satisfaire:
        * À l'arrêt: activer des commandes référencées par les produits les
          plus proches et créer jusqu'à K cartons (un par commande au départ).
        * En mouvement: se rendre au produit le plus proche encore nécessaire
          pour les commandes actives et remplir les cartons en respectant les
          capacités. Ouvrir de nouveaux cartons si besoin (dans la limite K).
        * Terminer la tournée si plus de demande/place, puis recommencer si
          nécessaire.
    - Générer la structure Output et écrire le fichier solution via Output.
    """
    debug_enabled = os.environ.get("SOLVER_DEBUG", "0") not in ("", "0")
    instance = Instance(filename)
    resolver = ShortestPathResolver(instance)
    prods = products_by_id(instance)
    start_loc = instance.graph["departing_depot"]
    end_loc = instance.graph["arrival_depot"]
    K = int(instance.nb_boxes_trolley or 6)
    capa = instance.capa_box or [10**9, 10**9]
    max_w = capa[0] if len(capa) > 0 else 10**9
    max_v = capa[1] if len(capa) > 1 else 10**9

    # 1) Liste des produits les plus proches du point de départ
    products_sorted = products_sorted_from_start(instance, resolver)

    # 2) États des commandes (quantités restantes à prélever)
    orders_state = build_orders_state(instance)

    # 3) Résultats à constituer
    tours_out: List[Tour] = []
    orders_out_map: Dict[int, int] = {}  # order_id -> total number of boxes over all tours

    # 4) Tant que toutes les commandes ne sont pas prêtes
    all_order_ids = list(orders_state.keys())
    tour_id = 1
    global_box_id = 1
    total_travelled_distance = 0
    total_crossed_locations = 0
    tour_logs: List[Dict[str, Any]] = []

    while True:
        # Stop si tout est fait
        if all(orders_state[oid].is_done() for oid in all_order_ids):
            break

        # Phase d'affectation à l'arrêt
        # Cas simple: si le mélange de commandes n'est pas autorisé, on
        # traite une commande à la fois (une par tournée). Cela garantit de
        # ne pas dépasser K colis pour une commande.
        active_orders: List[int] = []
        boxes: List[BoxState] = []
        if not instance.mixed_orders_allowed:
            # Une commande par tournée (pas de mélange), mais SANS limite globale de 6 par commande
            for oid in all_order_ids:
                if not orders_state[oid].is_done():
                    active_orders = [oid]
                    boxes = [BoxState(id=global_box_id, order_id=oid)]
                    global_box_id += 1
                    break
        else:
            # Si le mélange est autorisé, on pourrait réutiliser la version par proximité
            # et sélectionner plusieurs commandes, mais on garde ici une version simple:
            for oid in all_order_ids:
                if len(boxes) >= K:
                    break
                if not orders_state[oid].is_done():
                    active_orders.append(oid)
                    boxes.append(BoxState(id=global_box_id, order_id=oid))
                    global_box_id += 1

        if debug_enabled:
            print(
                f"[DEBUG] Tour {tour_id}: activation commandes={active_orders} (cartons initiaux={len(boxes)})"
            )

        # Sécurité: si aucune commande n'a pu être activée, on sort (ne devrait pas arriver)
        if not active_orders:
            break

        # Mouvement + remplissage (utilisé pour mixte et non-mixte)
        current_loc = start_loc
        travelled = 0
        crossed_locations = 0
        tour_active_orders = list(active_orders)

        # Helper: quick map order -> its boxes
        order_to_boxes: Dict[int, List[BoxState]] = {}
        for b in boxes:
            order_to_boxes.setdefault(b.order_id or -1, []).append(b)

        def can_place_in_any_box(oid: int, prod_id: int, qty_wanted: int) -> int:
            nonlocal global_box_id
            p = prods[prod_id]
            per_w, per_v = p["w"], p["v"]
            placed = 0
            # Try existing boxes of the order
            for b in order_to_boxes.get(oid, []):
                remain_w = max(0, max_w - b.weight)
                remain_v = max(0, max_v - b.volume)
                by_w = remain_w // max(1, per_w) if per_w > 0 else qty_wanted
                by_v = remain_v // max(1, per_v) if per_v > 0 else qty_wanted
                can_add = min(qty_wanted - placed, by_w, by_v)
                if can_add > 0:
                    b.add(per_w, per_v, prod_id, can_add)
                    placed += can_add
                if placed >= qty_wanted:
                    return placed
            # Open new boxes if needed, limited by trolley capacity K only
            while placed < qty_wanted and len(boxes) < K:
                new_b = BoxState(id=global_box_id, order_id=oid)
                global_box_id += 1
                boxes.append(new_b)
                order_to_boxes.setdefault(oid, []).append(new_b)
                remain_w = max_w
                remain_v = max_v
                by_w = remain_w // max(1, per_w) if per_w > 0 else qty_wanted
                by_v = remain_v // max(1, per_v) if per_v > 0 else qty_wanted
                can_add = min(qty_wanted - placed, by_w, by_v)
                if can_add <= 0:
                    break
                new_b.add(per_w, per_v, prod_id, can_add)
                placed += can_add
            return placed

        def can_placeable_units(oid: int, prod_id: int) -> int:
            p = prods[prod_id]
            per_w, per_v = p["w"], p["v"]
            total = 0
            for b in order_to_boxes.get(oid, []):
                remain_w = max(0, max_w - b.weight)
                remain_v = max(0, max_v - b.volume)
                by_w = remain_w // max(1, per_w) if per_w > 0 else 10**9
                by_v = remain_v // max(1, per_v) if per_v > 0 else 10**9
                total += max(0, min(by_w, by_v))
            available_slots = max(0, K - len(boxes))
            if available_slots > 0:
                by_w = (max_w // max(1, per_w)) if per_w > 0 else 10**9
                by_v = (max_v // max(1, per_v)) if per_v > 0 else 10**9
                cap = max(0, min(by_w, by_v))
                total += available_slots * cap
            return total

        # Tant qu'il reste de la demande pour les commandes actives et qu'on peut encore placer
        while True:
            needed: Set[int] = set()
            for oid in active_orders:
                if not orders_state[oid].is_done():
                    for pid, qty in orders_state[oid].remaining.items():
                        if qty > 0:
                            needed.add(pid)
            if not needed:
                break
            # Filtrer aux produits qui peuvent rentrer dans les cartons (existants + ouvrables jusqu'à K)
            placeable: Set[int] = set()
            for pid in needed:
                total_placeable = 0
                for oid in active_orders:
                    qty_need = orders_state[oid].remaining.get(pid, 0)
                    if qty_need > 0:
                        total_placeable += min(qty_need, can_placeable_units(oid, pid))
                if total_placeable > 0:
                    placeable.add(pid)
            if not placeable:
                break
            nxt = next_nearest_product(current_loc, placeable, prods, resolver)
            if nxt is None:
                break
            loc = prods[nxt]["id_loc"]
            dstep = resolver.dist(current_loc, loc)
            if dstep is None:
                break
            travelled += dstep
            current_loc = loc
            crossed_locations += 1
            if debug_enabled:
                print(
                    f"[DEBUG] Tour {tour_id}: sélection produit={nxt} distance+={dstep}"
                )
            # Remplir pour chaque commande active qui en a besoin (non-mixte => 1 commande)
            progress = 0
            for oid in active_orders:
                qty_need = orders_state[oid].remaining.get(nxt, 0)
                if qty_need <= 0:
                    continue
                placed = can_place_in_any_box(oid, nxt, qty_need)
                if placed > 0:
                    orders_state[oid].remaining[nxt] -= placed
                    progress += placed
                    if debug_enabled:
                        print(
                            f"    -> Commande {oid}: {placed}/{qty_need} unités chargées"
                        )
            if progress == 0:
                break

        # Retour au dépôt d'arrivée (si défini) pour terminer la tournée
        if current_loc is not None and end_loc is not None:
            dend = resolver.dist(current_loc, end_loc)
            if dend is not None:
                travelled += dend
                # On considère l'arrivée comme un point traversé/arrêté
                crossed_locations += 1

        # Matérialiser la tournée et les cartons pour la sortie
        boxes_out: List[Boxe] = []
        for b in boxes:
            if not b.products:
                # Skip empty boxes (unfilled due to capacity constraints)
                continue
            bps = [BoxeProduct(product_id=pid, quantity=qty) for pid, qty in b.products.items()]
            boxes_out.append(
                Boxe(
                    id=b.id,
                    order_id=b.order_id or -1,
                    weight=b.weight,
                    volume=b.volume,
                    boxe_products=bps,
                )
            )

        # Ajouter la tournée seulement si au moins un carton a été rempli
        if boxes_out:
            # Mettre à jour le total de cartons par commande (global) pour respecter la limite K
            per_order_new = {}
            for b in boxes_out:
                per_order_new[b.order_id] = per_order_new.get(b.order_id, 0) + 1
            # Appliquer l'incrément global
            for oid, cnt in per_order_new.items():
                orders_out_map[oid] = orders_out_map.get(oid, 0) + cnt

            tours_out.append(Tour(id=tour_id, boxes=boxes_out))
            tour_id += 1
            total_travelled_distance += travelled
            total_crossed_locations += crossed_locations

            total_box_capacity_w = len(boxes_out) * max(1, max_w)
            total_box_capacity_v = len(boxes_out) * max(1, max_v)
            weight_fill_pct = (
                int(round(sum(b.weight for b in boxes_out) * 100 / total_box_capacity_w))
                if total_box_capacity_w
                else 0
            )
            volume_fill_pct = (
                int(round(sum(b.volume for b in boxes_out) * 100 / total_box_capacity_v))
                if total_box_capacity_v
                else 0
            )
            tour_logs.append(
                {
                    "tour_id": tour_id - 1,
                    "orders": tour_active_orders,
                    "boxes": len(boxes_out),
                    "travelled_distance": travelled,
                    "crossed_locations": crossed_locations,
                    "weight_fill_pct": weight_fill_pct,
                    "volume_fill_pct": volume_fill_pct,
                }
            )
            if debug_enabled:
                print(
                    "[DEBUG] Tour {tid}: distance={dist} traverses={cross} cartons={boxes} "
                    "remplissage={weight}%/{volume}%".format(
                        tid=tour_id - 1,
                        dist=travelled,
                        cross=crossed_locations,
                        boxes=len(boxes_out),
                        weight=weight_fill_pct,
                        volume=volume_fill_pct,
                    )
                )

        # Loop continues until all orders done; if some active orders were not fully satisfied due to capacity, they'll be picked in next tour

    # Calculer quelques statistiques synthétiques
    # Note: on ne mémorise pas ici la distance parcourue par tournée; on
    # pourrait l'accumuler si nécessaire. Le checker lit surtout la structure
    # des tournées/cartons. On calcule cependant des pourcentages moyen de
    # remplissage (poids/volume) pour information.

    all_boxes: List[Boxe] = [b for t in tours_out for b in t.boxes]
    if all_boxes:
        avg_weight = int(round(sum(b.weight for b in all_boxes) * 100 / (len(all_boxes) * max(1, max_w))))
        avg_volume = int(round(sum(b.volume for b in all_boxes) * 100 / (len(all_boxes) * max(1, max_v))))
    else:
        avg_weight = 0
        avg_volume = 0

    log_path = os.environ.get("SOLVER_TOUR_LOG")
    if log_path and tour_logs:
        try:
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(tour_logs, fh, indent=2)
        except OSError as exc:
            print(f"[WARN] Impossible d'écrire le journal des tournées '{log_path}': {exc}")

    # Section commandes pour la sortie (nombre total de cartons par commande)
    orders_out: List[OutOrder] = [OutOrder(id=oid, nbr_boxes=orders_out_map.get(oid, 0)) for oid in sorted(orders_out_map)]

    # travelled_distance / crossed_locations: mis à 0 pour le moment; on peut
    # les affiner si nécessaire par la suite.
    out = Output(
        filename=filename,
        travelled_distance=total_travelled_distance,
        crossed_locations=total_crossed_locations,
        avg_weight=avg_weight,
        avg_volume=avg_volume,
        tours=tours_out,
        orders=orders_out,
    )
    return out


__all__ = ["solve_instance"]
