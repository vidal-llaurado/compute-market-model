"""Solve and price a synthetic compute contract."""
from compute_market.examples import synthetic_case
from compute_market.optimizer import solve
from compute_market.pricing import reservation_price

portfolio, book, offered = synthetic_case(depth=2)
result = solve(portfolio, book)
if result.status != 'optimal':
    raise RuntimeError(result.message)
quote = reservation_price(portfolio, offered, book=book)
print('Synthetic reservation premium:', quote.amount, quote.kind)
print('Value comparison:', quote.base_value, quote.accepted_value)
print('Numerical checks:', result.audit)
