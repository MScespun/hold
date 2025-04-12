from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import List, Any
import string
import jsonpickle
import numpy as np
import math
import json


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )
        # Truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit.
        max_item_length = (self.max_log_length - base_length) // 3
        ''' 
        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )
        '''
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])
        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]
        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]
        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."


logger = Logger()


class Trader:
    def __init__(self):
        # Existing trackers for KELP
        self.kelp_prices = []
        self.kelp_vwap = []
        # NEW: Trackers for SQUID_INK
        self.squid_prices = []  # For SQUID_INK mid-price history
        self.squid_vwap = []  # For SQUID_INK VWAP values

    def rainforest_orders(self, order_depth: OrderDepth, fair_value: int, width: int, position: int,
                          position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        # Use fallback values if the filtered list is empty.
        sell_filtered = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
        buy_filtered = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        baaf = min(sell_filtered) if sell_filtered else fair_value + 2
        bbbf = max(buy_filtered) if buy_filtered else fair_value - 2

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if best_ask < fair_value:
                quantity = min(best_ask_amount, position_limit - position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_ask, quantity))
                    buy_order_volume += quantity

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid > fair_value:
                quantity = min(best_bid_amount, position_limit + position)
                if quantity > 0:
                    orders.append(Order("RAINFOREST_RESIN", best_bid, -quantity))
                    sell_order_volume += quantity

        buy_order_volume, sell_order_volume = self.clear_position_order(
            orders, order_depth, position, position_limit, "RAINFOREST_RESIN",
            buy_order_volume, sell_order_volume, fair_value, 1
        )

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(round(bbbf + 1)), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(round(baaf - 1)), -sell_quantity))

        return orders
    
    def clear_position_order(self, orders: List[Order], order_depth: OrderDepth, position: int, position_limit: int,
                             product: str, buy_order_volume: int, sell_order_volume: int, fair_value: float,
                             width: int) -> tuple[int, int]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair = int(round(fair_value))
        fair_for_bid = int(math.floor(fair_value))
        fair_for_ask = int(math.ceil(fair_value))

        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def kelp_fair_value(self, order_depth: OrderDepth, method="mid_price", min_vol=0) -> int:
        if method == "mid_price":
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = int(round((best_ask + best_bid) / 2))
            return mid_price
        elif method == "mid_price_with_vol_filter":
            if (len([price for price in order_depth.sell_orders.keys() if
                     abs(order_depth.sell_orders[price]) >= min_vol]) == 0 or
                    len([price for price in order_depth.buy_orders.keys() if
                         abs(order_depth.buy_orders[price]) >= min_vol]) == 0):
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = int(round((best_ask + best_bid) / 2))
                return mid_price
            else:
                best_ask = min([price for price in order_depth.sell_orders.keys() if
                                abs(order_depth.sell_orders[price]) >= min_vol])
                best_bid = max(
                    [price for price in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[price]) >= min_vol])
                mid_price = int(round((best_ask + best_bid) / 2))
            return mid_price

    def kelp_orders(self, order_depth: OrderDepth, timespan: int, width: float, kelp_take_width: float, position: int,
                    position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if
                            abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if
                            abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = int(round((mm_ask + mm_bid) / 2))
            self.kelp_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] +
                    best_ask * order_depth.buy_orders[best_bid]) / volume
            self.kelp_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.kelp_vwap) > timespan:
                self.kelp_vwap.pop(0)
            if len(self.kelp_prices) > timespan:
                self.kelp_prices.pop(0)

            fair_value = mmmid_price

            if best_ask <= fair_value - kelp_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + kelp_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("KELP", best_bid, -quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "KELP", buy_order_volume, sell_order_volume, fair_value,
                2
            )

            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("KELP", int(round(bbbf + 1)), buy_quantity))  # Force int price

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("KELP", int(round(baaf - 1)), -sell_quantity))  # Force int price

        return orders

    # NEW: SQUID strategy: Essentially the same as kelp_orders with the symbol changed to "SQUID_INK"
    def squid_ink_orders(self, order_depth: OrderDepth, timespan: int, width: float, squid_take_width: float,
                         position: int, position_limit: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys() if
                            abs(order_depth.sell_orders[price]) >= 15]
            filtered_bid = [price for price in order_depth.buy_orders.keys() if
                            abs(order_depth.buy_orders[price]) >= 15]
            mm_ask = min(filtered_ask) if filtered_ask else best_ask
            mm_bid = max(filtered_bid) if filtered_bid else best_bid

            mmmid_price = int(round((mm_ask + mm_bid) / 2))
            self.squid_prices.append(mmmid_price)

            volume = -1 * order_depth.sell_orders[best_ask] + order_depth.buy_orders[best_bid]
            vwap = (best_bid * (-1) * order_depth.sell_orders[best_ask] +
                    best_ask * order_depth.buy_orders[best_bid]) / volume
            self.squid_vwap.append({"vol": volume, "vwap": vwap})

            if len(self.squid_vwap) > timespan:
                self.squid_vwap.pop(0)
            if len(self.squid_prices) > timespan:
                self.squid_prices.pop(0)

            fair_value = mmmid_price

            if best_ask <= fair_value - squid_take_width:
                ask_amount = -1 * order_depth.sell_orders[best_ask]
                if ask_amount <= 20:
                    quantity = min(ask_amount, position_limit - position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_ask, quantity))
                        buy_order_volume += quantity
            if best_bid >= fair_value + squid_take_width:
                bid_amount = order_depth.buy_orders[best_bid]
                if bid_amount <= 20:
                    quantity = min(bid_amount, position_limit + position)
                    if quantity > 0:
                        orders.append(Order("SQUID_INK", best_bid, -quantity))
                        sell_order_volume += quantity

            buy_order_volume, sell_order_volume = self.clear_position_order(
                orders, order_depth, position, position_limit, "SQUID_INK", buy_order_volume, sell_order_volume,
                fair_value, 2
            )

            aaf = [price for price in order_depth.sell_orders.keys() if price > fair_value + 1]
            bbf = [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
            baaf = min(aaf) if aaf else fair_value + 2
            bbbf = max(bbf) if bbf else fair_value - 2

            buy_quantity = position_limit - (position + buy_order_volume)
            if buy_quantity > 0:
                orders.append(Order("SQUID_INK", int(round(bbbf + 1)), buy_quantity))  # Force int price

            sell_quantity = position_limit + (position - sell_order_volume)
            if sell_quantity > 0:
                orders.append(Order("SQUID_INK", int(round(baaf - 1)), -sell_quantity))  # Force int price

        return orders

    def comb_1_orders(self, order_depth_croissants: OrderDepth, order_depth_jams: OrderDepth,
                 order_depth_basket1: OrderDepth, fair_value: int,
                      fair_value_croissant: int, fair_value_jams: int, fair_value_basket1: int, width: int, position_croissant: int, position_limit_croissant: int,
                      position_jams: int, position_limit_jams: int,
                      position_basket1: int, position_limit_basket1: int) -> List[Order]:
        orders: List[Order] = []
        buy_order_volume_croissant = 0
        sell_order_volume_croissant = 0
        buy_order_volume_jams = 0
        sell_order_volume_jams = 0
        
        buy_order_volume_basket1 = 0
        sell_order_volume_basket1 = 0

        ########MODIFICARE QUESTA PARTE
        sell_filtered = [price for price in order_depth_croissants.sell_orders.keys() if price > fair_value_croissant + 1]
        buy_filtered = [price for price in order_depth_croissants.buy_orders.keys() if price < fair_value_croissant - 1]
        baaf_croissant = min(sell_filtered) if sell_filtered else fair_value_croissant + 2
        bbbf_croissant = max(buy_filtered) if buy_filtered else fair_value_croissant - 2

        sell_filtered = [price for price in order_depth_jams.sell_orders.keys() if price > fair_value_jams + 1]
        buy_filtered = [price for price in order_depth_jams.buy_orders.keys() if price < fair_value_jams - 1]
        baaf_jams = min(sell_filtered) if sell_filtered else fair_value_jams + 2
        bbbf_jams = max(buy_filtered) if buy_filtered else fair_value_jams - 2

        

        sell_filtered = [price for price in order_depth_basket1.sell_orders.keys() if price > fair_value_basket1 + 1]
        buy_filtered = [price for price in order_depth_basket1.buy_orders.keys() if price < fair_value_basket1 - 1]
        baaf_basket1 = min(sell_filtered) if sell_filtered else fair_value_basket1 + 2
        bbbf_basket1 = max(buy_filtered) if buy_filtered else fair_value_basket1 - 2

  
        value_jams = -0.234596382214451
        value_basket1 = 0.09897279928482106
        value_croissants = -1.0
        

        best_ask_croissants = min(
            order_depth_croissants.sell_orders.keys()) if order_depth_croissants.sell_orders else float('inf')
        best_bid_croissants = max(
            order_depth_croissants.buy_orders.keys()) if order_depth_croissants.buy_orders else float('-inf')
        best_ask_jams = min(order_depth_jams.sell_orders.keys()) if order_depth_jams.sell_orders else float('inf')
        best_bid_jams = max(order_depth_jams.buy_orders.keys()) if order_depth_jams.buy_orders else float('-inf')
        
        best_ask_basket1 = min(order_depth_basket1.sell_orders.keys()) if order_depth_basket1.sell_orders else float(
            'inf')
        best_bid_basket1 = max(order_depth_basket1.buy_orders.keys()) if order_depth_basket1.buy_orders else float(
            '-inf')

        best_ask_croissants_volume = -1 * order_depth_croissants.sell_orders[
            best_ask_croissants] if best_ask_croissants in order_depth_croissants.sell_orders else 0
        best_ask_jams_volume = -1 * order_depth_jams.sell_orders[
            best_ask_jams] if best_ask_jams in order_depth_jams.sell_orders else 0
        
        best_ask_basket1_volume = -1 * order_depth_basket1.sell_orders[
            best_ask_basket1] if best_ask_basket1 in order_depth_basket1.sell_orders else 0
    
        best_bid_croissants_volume = order_depth_croissants.buy_orders[
            best_bid_croissants] if best_bid_croissants in order_depth_croissants.buy_orders else 0
        best_bid_jams_volume = order_depth_jams.buy_orders[
            best_bid_jams] if best_bid_jams in order_depth_jams.buy_orders else 0
       
        best_bid_basket1_volume = order_depth_basket1.buy_orders[
            best_bid_basket1] if best_bid_basket1 in order_depth_basket1.buy_orders else 0
      
        if value_croissants < 0:
            temp = best_ask_croissants
            best_ask_croissants = best_bid_croissants
            best_bid_croissants = temp
            temp2 = best_ask_croissants_volume
            best_ask_croissants_volume = best_bid_croissants_volume
            best_bid_croissants_volume = temp2

        if value_jams < 0:
            temp = best_ask_jams
            best_ask_jams = best_bid_jams
            best_bid_jams = temp
            temp2 = best_ask_jams_volume
            best_ask_jams_volume = best_bid_jams_volume
            best_bid_jams_volume = temp2

      

        if value_basket1 < 0:
            temp = best_ask_basket1
            best_ask_basket1 = best_bid_basket1
            best_bid_basket1 = temp
            temp2 = best_ask_basket1_volume
            best_ask_basket1_volume = best_bid_basket1_volume
            best_bid_basket1_volume = temp2
   
        best_ask = value_croissants * best_ask_croissants + value_jams * best_ask_jams +  value_basket1 * best_ask_basket1
        best_bid = value_croissants * best_bid_croissants + value_jams * best_bid_jams +  value_basket1 * best_bid_basket1
        best_ask_volume = math.floor(abs(min(best_ask_croissants_volume / value_croissants, best_ask_jams_volume / value_jams,
                                 best_ask_basket1_volume / value_basket1,
                                key=abs)))
        best_bid_volume = math.floor(abs(min(best_bid_croissants_volume / value_croissants, best_bid_jams_volume / value_jams,
                                 best_bid_basket1_volume / value_basket1,
                                key=abs)))
        
        order_depth = OrderDepth()
        order_depth.sell_orders = {int(best_ask):  best_ask_volume}
        order_depth.buy_orders = {int(best_bid): best_bid_volume}
        
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())

            value_volumes = abs(min(best_ask_croissants_volume / value_croissants, best_ask_jams_volume / value_jams,
                                best_ask_basket1_volume / value_basket1,
                                key=abs))
            best_ask_amount_croissants = math.floor(abs(value_volumes * value_croissants))
            best_ask_amount_jams = math.floor(abs(value_volumes * value_jams))
          
            best_ask_amount_basket1 = math.floor(abs(value_volumes * value_basket1))
            if best_ask < fair_value-1:

               
                quantity_croissants = min(best_ask_amount_croissants, position_limit_croissant - position_croissant)
                quantity_jams = min(best_ask_amount_jams, position_limit_jams - position_jams)
               
                quantity_basket1 = min(best_ask_amount_basket1, position_limit_basket1 - position_basket1)
                if quantity_croissants > 0 and quantity_jams > 0  and quantity_basket1 > 0:
                    orders.append(Order("CROISSANTS", best_ask_croissants, quantity_croissants * int(np.sign(value_croissants))))
                    orders.append(Order("JAMS", best_ask_jams,quantity_jams * int(np.sign(value_jams))))
                    orders.append(Order("PICNIC_BASKET1", best_ask_basket1, quantity_basket1 * int(np.sign(value_basket1))))
                    buy_order_volume_croissant += quantity_croissants
                    buy_order_volume_jams += quantity_jams
                    buy_order_volume_basket1 += quantity_basket1

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())

            value_volumes = math.floor(abs(min(best_bid_croissants_volume / value_croissants, best_bid_jams_volume / value_jams,
                                best_bid_basket1_volume / value_basket1,
                                key=abs)))
            best_bid_amount_croissants = math.floor(abs(value_volumes * value_croissants))
            best_bid_amount_jams = math.floor(abs(value_volumes * value_jams))
            best_bid_amount_basket1 = math.floor(abs(value_volumes * value_basket1))
            if best_bid > fair_value+1:
                quantity_croissants = min(best_bid_amount_croissants, position_limit_croissant + position_croissant)
                quantity_jams = min(best_bid_amount_jams, position_limit_jams + position_jams)
                quantity_basket1 = min(best_bid_amount_basket1, position_limit_basket1 + position_basket1)
                if quantity_croissants > 0 and quantity_jams > 0 and quantity_basket1 > 0:
                    orders.append(Order("CROISSANTS", best_bid_croissants, -1*quantity_croissants * int(np.sign(value_croissants))))
                    orders.append(Order("JAMS", best_bid_jams, -1*quantity_jams * int(np.sign(value_jams))))
                    orders.append(Order("PICNIC_BASKET1", best_bid_basket1, -1*quantity_basket1 * int(np.sign(value_basket1))))
                    sell_order_volume_croissant += quantity_croissants
                    sell_order_volume_jams += quantity_jams
                    sell_order_volume_basket1 += quantity_basket1
                            # --- INIZIO CLEARING DELLA COMBINAZIONE AGGREGATA ---
        # Clearing per CROISSANTS
        position_after_croissant = position_croissant + (buy_order_volume_croissant - sell_order_volume_croissant)
        fair_croissant = fair_value_croissant  # Il fair value specifico per CROISSANTS
        fair_for_bid_croissant = int(math.floor(fair_croissant))
        fair_for_ask_croissant = int(math.ceil(fair_croissant))
        buy_qty_croissant = position_limit_croissant - (position_croissant + buy_order_volume_croissant)
        sell_qty_croissant = position_limit_croissant + (position_croissant - sell_order_volume_croissant)
        if position_after_croissant > 0:
            # Se si è long in eccesso, proviamo a vendere (chiudere posizioni) al prezzo pari a fair_for_ask
            if fair_for_ask_croissant in order_depth_croissants.buy_orders.keys():
                clear_qty = min(order_depth_croissants.buy_orders[fair_for_ask_croissant], position_after_croissant)
                sent_qty = min(sell_qty_croissant, clear_qty)
                orders.append(Order("CROISSANTS", fair_for_ask_croissant, -abs(sent_qty)))
                sell_order_volume_croissant += abs(sent_qty)
        elif position_after_croissant < 0:
            # Se si è short in eccesso, proviamo a comprare al prezzo pari a fair_for_bid
            if fair_for_bid_croissant in order_depth_croissants.sell_orders.keys():
                clear_qty = min(abs(order_depth_croissants.sell_orders[fair_for_bid_croissant]), abs(position_after_croissant))
                sent_qty = min(buy_qty_croissant, clear_qty)
                orders.append(Order("CROISSANTS", fair_for_bid_croissant, abs(sent_qty)))
                buy_order_volume_croissants += abs(sent_qty)

        # Clearing per JAMS
        position_after_jams = position_jams + (buy_order_volume_jams - sell_order_volume_jams)
        fair_jams = fair_value_jams  # Il fair value per JAMS
        fair_for_bid_jams = int(math.floor(fair_jams))
        fair_for_ask_jams = int(math.ceil(fair_jams))
        buy_qty_jams = position_limit_jams - (position_jams + buy_order_volume_jams)
        sell_qty_jams = position_limit_jams + (position_jams - sell_order_volume_jams)
        if position_after_jams > 0:
            if fair_for_ask_jams in order_depth_jams.buy_orders.keys():
                clear_qty = min(order_depth_jams.buy_orders[fair_for_ask_jams], position_after_jams)
                sent_qty = min(sell_qty_jams, clear_qty)
                orders.append(Order("JAMS", fair_for_ask_jams, -abs(sent_qty)))
                sell_order_volume_jams += abs(sent_qty)
        elif position_after_jams < 0:
            if fair_for_bid_jams in order_depth_jams.sell_orders.keys():
                clear_qty = min(abs(order_depth_jams.sell_orders[fair_for_bid_jams]), abs(position_after_jams))
                sent_qty = min(buy_qty_jams, clear_qty)
                orders.append(Order("JAMS", fair_for_bid_jams, abs(sent_qty)))
                buy_order_volume_jams += abs(sent_qty)

        # Clearing per PICNIC_BASKET1
        position_after_basket1 = position_basket1 + (buy_order_volume_basket1 - sell_order_volume_basket1)
        fair_basket1 = fair_value_basket1  # Il fair value per PICNIC_BASKET1
        fair_for_bid_basket1 = int(math.floor(fair_basket1))
        fair_for_ask_basket1 = int(math.ceil(fair_basket1))
        buy_qty_basket1 = position_limit_basket1 - (position_basket1 + buy_order_volume_basket1)
        sell_qty_basket1 = position_limit_basket1 + (position_basket1 - sell_order_volume_basket1)
        if position_after_basket1 > 0:
            if fair_for_ask_basket1 in order_depth_basket1.buy_orders.keys():
                clear_qty = min(order_depth_basket1.buy_orders[fair_for_ask_basket1], position_after_basket1)
                sent_qty = min(sell_qty_basket1, clear_qty)
                orders.append(Order("PICNIC_BASKET1", fair_for_ask_basket1, -abs(sent_qty)))
                sell_order_volume_basket1 += abs(sent_qty)
        elif position_after_basket1 < 0:
            if fair_for_bid_basket1 in order_depth_basket1.sell_orders.keys():
                clear_qty = min(abs(order_depth_basket1.sell_orders[fair_for_bid_basket1]), abs(position_after_basket1))
                sent_qty = min(buy_qty_basket1, clear_qty)
                orders.append(Order("PICNIC_BASKET1", fair_for_bid_basket1, abs(sent_qty)))
                buy_order_volume_basket1 += abs(sent_qty)
        # --- FINE CLEARING DELLA COMBINAZIONE AGGREGATA ---

                    
  

        return orders  


        
        '''  
        position_after_take =  np.sign(value_croissants)*min(position_croissant / value_croissants, position_jams/ value_jams,
                                position_djembes / value_djembes, position_basket1 / value_basket1,
                                key=abs)
        buy_order_volume = min(buy_order_volume_croissant/value_croissants, buy_order_volume_jams/value_jams,
                                buy_order_volume_djembes/value_djembes, buy_order_volume_basket1/value_basket1,
                                key=abs)
        sell_order_volume = min(sell_order_volume_croissant/value_croissants, sell_order_volume_jams/value_jams,
                                sell_order_volume_djembes/value_djembes, sell_order_volume_basket1/value_basket1,
                                key=abs)
        position = position_after_take + sell_order_volume-buy_order_volume
        
        fair_for_bid = fair_value
        fair_for_ask = fair_value+1
        
        position_limit = min(position_limit_croissant / value_croissants, position_limit_jams/ value_jams,
                                position_limit_djembes / value_djembes, position_limit_basket1 / value_basket1,
                                key=abs)
        
        buy_quantity = position_limit - (position + buy_order_volume)
        sell_quantity = position_limit + (position - sell_order_volume)

        if position_after_take > 0:
            if fair_for_ask in order_depth.buy_orders.keys():
                clear_quantity = min(order_depth.buy_orders[fair_for_ask], position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            if fair_for_bid in order_depth.sell_orders.keys():
                clear_quantity = min(abs(order_depth.sell_orders[fair_for_bid]), abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)
        

        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(round(bbbf + 1)), buy_quantity))

        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order("RAINFOREST_RESIN", int(round(baaf - 1)), -sell_quantity))

        return orders
        '''
#FAIRVALUE SINGOLO??????????
    '''
        buy_order_volume_croissant, sell_order_volume_croissant = self.clear_position_order(
            orders, order_depth, position_croissant, position_limit_croissant, "CROISSANT",
            buy_order_volume_croissant, sell_order_volume_croissant, fair_value_croissant, 1
        )

        buy_order_volume_jams, sell_order_volume_jams = self.clear_position_order(
            orders, order_depth, position_jams, position_limit_jams, "JAMS",
            buy_order_volume_jams, sell_order_volume_jams, fair_value_jams, 1
        )
        buy_order_volume_djembes, sell_order_volume_djembes = self.clear_position_order(
            orders, order_depth, position_djembes, position_limit_djembes, "DJEMBES",
            buy_order_volume_djembes, sell_order_volume_djembes, fair_value_djembes, 1
        )
        buy_order_volume_basket1, sell_order_volume_basket1 = self.clear_position_order(
            orders, order_depth, position_basket1, position_limit_basket1, "BASKET1",
            buy_order_volume_basket1, sell_order_volume_basket1, fair_value_basket1, 1
        )

        buy_quantity_croissant = position_limit_croissant - (position_croissant + buy_order_volume_croissant)
        if buy_quantity_croissant > 0:
            orders.append(Order("CROISSANT", int(round(bbbf_croissant + 1)), buy_quantity_croissant))

        sell_quantity_croissant = position_limit_croissant + (position_croissant - sell_order_volume_croissant)
        if sell_quantity_croissant > 0:
            orders.append(Order("CROISSANT", int(round(baaf_croissant - 1)), -sell_quantity_croissant))

        buy_quantity_jams = position_limit_jams - (position_jams + buy_order_volume_jams)
        if buy_quantity_jams > 0:
            orders.append(Order("JAMS", int(round(bbbf_jams + 1)), buy_quantity_jams))

        sell_quantity_jams = position_limit_jams + (position_jams - sell_order_volume_jams)
        if sell_quantity_jams > 0:
            orders.append(Order("JAMS", int(round(baaf_jams - 1)), -sell_quantity_jams))

        buy_quantity_djembes = position_limit_djembes - (position_djembes + buy_order_volume_djembes)
        if buy_quantity_djembes > 0:
            orders.append(Order("DJEMBES", int(round(bbbf_djembes + 1)), buy_quantity_djembes))

        sell_quantity_djembes = position_limit_djembes + (position_djembes - sell_order_volume_djembes)
        if sell_quantity_djembes > 0:
            orders.append(Order("DJEMBES", int(round(baaf_djembes - 1)), -sell_quantity_djembes))

        buy_quantity_basket1 = position_limit_basket1 - (position_basket1 + buy_order_volume_basket1)
        if buy_quantity_basket1 > 0:
            orders.append(Order("BASKET1", int(round(bbbf_basket1 + 1)), buy_quantity_basket1))

        sell_quantity_basket1 = position_limit_basket1 + (position_basket1 - sell_order_volume_basket1)
        if sell_quantity_basket1 > 0:
            orders.append(Order("BASKET1", int(round(baaf_basket1 - 1)), -sell_quantity_basket1))

        return orders

        ##########################
   '''
    def run(self, state: TradingState) -> tuple[dict[Symbol, list[Order]], int, str]:
        result = {}

        rainforest_fair_value = 10000  # Participant should calculate this value
        rainforest_width = 2
        rainforest_position_limit = 50
        position_limit_croissant = 250
        position_limit_jams = 350
        position_limit_djembes = 60
        position_limit_basket1 = 60
        position_limit_basket2 = 100

        if "RAINFOREST_RESIN" in state.order_depths:
            rainforest_position = state.position.get("RAINFOREST_RESIN", 0)
            rainforest_orders = self.rainforest_orders(
                state.order_depths["RAINFOREST_RESIN"],
                rainforest_fair_value,
                rainforest_width,
                rainforest_position,
                rainforest_position_limit
            )
            #result["RAINFOREST_RESIN"] = rainforest_orders
            e=3
        if "CROISSANTS" in state.order_depths and "JAMS" in state.order_depths and "PICNIC_BASKET2" in state.order_depths and "PICNIC_BASKET1" in state.order_depths:
            
            order_depth_croissants = state.order_depths["CROISSANTS"]
            order_depth_jams = state.order_depths["JAMS"]
            order_depth_basket1 = state.order_depths["PICNIC_BASKET1"]
            position_croissant = state.position.get("CROISSANTS", 0)
            position_jams = state.position.get("JAMS", 0)
            position_basket1 = state.position.get("PICNIC_BASKET1", 0)
            ordini_fun = self.comb_1_orders(order_depth_croissants, order_depth_jams,
                       order_depth_basket1, 0,
                      0,0,0, 0, position_croissant, position_limit_croissant,
                      position_jams, position_limit_jams,  
                      position_basket1, position_limit_basket1)
         
            
            
            for order in ordini_fun:
                if order.symbol not in result:
                    result[order.symbol] = [order]
                else:   
                    result[order.symbol].append(order)
 
        
        kelp_make_width = 3.5
        kelp_take_width = 1
        kelp_position_limit = 50
        kelp_timespan = 10

        if "KELP" in state.order_depths:
            kelp_position = state.position.get("KELP", 0)
            kelp_orders = self.kelp_orders(
                state.order_depths["KELP"],
                kelp_timespan,
                kelp_make_width,
                kelp_take_width,
                kelp_position,
                kelp_position_limit
            )
            #result["KELP"] = kelp_orders
            g=3
        squid_make_width = 3.5
        squid_take_width = 1
        squid_position_limit = 50
        squid_timespan = 10

        if "SQUID_INK" in state.order_depths:
            squid_position = state.position.get("SQUID_INK", 0)
            squid_orders = self.squid_ink_orders(
                state.order_depths["SQUID_INK"],
                squid_timespan,
                squid_make_width,
                squid_take_width,
                squid_position,
                squid_position_limit
            )
            #result["SQUID_INK"] = squid_orders
            t=3
        traderData = jsonpickle.encode({
            "kelp_prices": self.kelp_prices,
            "kelp_vwap": self.kelp_vwap,
            "squid_prices": self.squid_prices,
            "squid_vwap": self.squid_vwap
        })

        conversions = 1
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, traderData