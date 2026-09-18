"""
app/services/limitation_service.py
==================================
LegalMind AI — Limitation Period Calculator
Rule-based engine for calculating filing deadlines under the Limitation Act, 1963.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List

class LimitationService:
    # Basic lookup table for Limitation Act, 1963 Schedule
    LIMITATION_DB = {
        "money_suit": {"period_years": 3, "period_days": 0, "article": "Art. 18-55"},
        "specific_performance": {"period_years": 3, "period_days": 0, "article": "Art. 54"},
        "declaration": {"period_years": 3, "period_days": 0, "article": "Art. 56-58"},
        "possession_immovable": {"period_years": 12, "period_days": 0, "article": "Art. 64-65"},
        "execution_decree": {"period_years": 12, "period_days": 0, "article": "Art. 136"},
        "appeal_high_court": {"period_years": 0, "period_days": 90, "article": "Art. 116(a)"},
        "appeal_other_court": {"period_years": 0, "period_days": 30, "article": "Art. 116(b)"},
        "leave_to_appeal_sc": {"period_years": 0, "period_days": 90, "article": "Art. 133"},
        "criminal_appeal_death": {"period_years": 0, "period_days": 30, "article": "Art. 115(a)"},
        "criminal_appeal_other": {"period_years": 0, "period_days": 60, "article": "Art. 115(b)"},
        "revision": {"period_years": 0, "period_days": 90, "article": "Art. 131"},
        "consumer_complaint": {"period_years": 2, "period_days": 0, "article": "S. 69 CPA 2019"},
        "cheque_bounce_notice": {"period_years": 0, "period_days": 30, "article": "S. 138(b) NI Act"},
        "cheque_bounce_complaint": {"period_years": 0, "period_days": 30, "article": "S. 142(b) NI Act"},
    }

    @staticmethod
    def calculate_limitation(
        suit_type: str, 
        cause_of_action_date: str, 
        exclude_days: int = 0
    ) -> Dict[str, Any]:
        
        try:
            for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d"):
                try:
                    start_date = datetime.strptime(cause_of_action_date, fmt)
                    break
                except ValueError:
                    continue
            else:
                raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        except Exception as e:
             raise ValueError(f"Date parsing error: {e}")

        rule = LimitationService.LIMITATION_DB.get(suit_type)
        if not rule:
            rule = {"period_years": 3, "period_days": 0, "article": "Residuary (Art. 113 / 137)"} # Default residuary
            
        # Calculate expiry date
        expiry_date = start_date
        if rule["period_years"] > 0:
            # Simple approximation for years
            try:
                expiry_date = expiry_date.replace(year=expiry_date.year + rule["period_years"])
            except ValueError:
                # Handle leap year Feb 29
                expiry_date = expiry_date + (timedelta(days=365) * rule["period_years"])
                
        if rule["period_days"] > 0:
            expiry_date = expiry_date + timedelta(days=rule["period_days"])
            
        # Exclusions (e.g. time taken to obtain certified copy, Section 12)
        if exclude_days > 0:
            expiry_date = expiry_date + timedelta(days=exclude_days)
            
        # Check if expired
        today = datetime.now()
        is_expired = today > expiry_date
        
        diff = expiry_date - today
        days_remaining = diff.days if not is_expired else 0
        days_delayed = abs(diff.days) if is_expired else 0

        condonation_advice = None
        if is_expired:
            if "appeal" in suit_type or suit_type in ["revision", "cheque_bounce_complaint"]:
                condonation_advice = "Delay may be condoned under Section 5 of the Limitation Act upon showing 'sufficient cause'. Draft a Section 5 application."
            elif suit_type in ["money_suit", "specific_performance", "declaration", "possession_immovable"]:
                condonation_advice = "Section 5 does NOT apply to original suits. The suit is likely time-barred unless saved by acknowledgment (S.18) or fraud (S.17)."

        return {
            "suit_type": suit_type,
            "cause_of_action_date": start_date.strftime("%Y-%m-%d"),
            "applicable_provision": rule["article"],
            "period_statute": f"{rule['period_years']} years, {rule['period_days']} days",
            "excluded_days": exclude_days,
            "expiry_date": expiry_date.strftime("%Y-%m-%d"),
            "status": "EXPIRED" if is_expired else "WITHIN LIMITATION",
            "days_remaining": days_remaining,
            "days_delayed": days_delayed,
            "condonation_advice": condonation_advice
        }
