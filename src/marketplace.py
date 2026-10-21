"""
Marketplace Integration Module for Car Spare Parts Recognition.
Maps predicted spare-part class labels to e-commerce catalog metadata.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

try:
    from .config import MARKETPLACE_CATALOG_FILE
except ImportError:
    from config import MARKETPLACE_CATALOG_FILE


class MarketplaceCatalog:
    """Catalog manager providing product details for classified spare parts."""

    def __init__(self, catalog_path: Optional[Path] = None):
        self.catalog_path = Path(catalog_path) if catalog_path else MARKETPLACE_CATALOG_FILE
        self._catalog_data = self._load_catalog()

    def _load_catalog(self) -> Dict[str, Any]:
        if not self.catalog_path.exists():
            return {}
        with open(self.catalog_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_part_info(self, class_name: str) -> Dict[str, Any]:
        """Fetch full marketplace metadata for a given class label.
        
        Args:
            class_name: Recognized class string (e.g. 'brake_pad')
            
        Returns:
            Dictionary with category, compatibility, brands, price range, and sellers.
        """
        # normalize string
        normalized = class_name.lower().strip().replace(" ", "_")
        if normalized in self._catalog_data:
            return self._catalog_data[normalized]
        
        # Fallback default template if class is unfamiliar
        return {
            "part_id": f"SP-UNKNOWN-{normalized.upper()[:3]}",
            "display_name": class_name.replace("_", " ").title(),
            "category": "General Automotive Parts",
            "description": "Standard replacement automotive spare part.",
            "compatible_vehicles": ["Vehicle compatibility check required"],
            "sample_brands": ["OEM Generic"],
            "price_range_usd": {"min": 15.00, "max": 150.00, "average": 45.00},
            "availability": "Contact Vendor",
            "warranty_months": 12,
            "product_image_placeholder": "assets/products/placeholder.jpg",
            "sellers": []
        }

    def format_marketplace_card(self, class_name: str, confidence: float) -> str:
        """Render a clean marketplace product summary without price, availability, brands, or compatibility."""
        info = self.get_part_info(class_name)
        card = [
            "=" * 60,
            "  MARKETPLACE PRODUCT DETAILS",
            "=" * 60,
            f"  Product Name:      {info.get('display_name')}",
            f"  AI Detection:      {class_name} ({confidence * 100:.1f}% confidence)",
            f"  Category:          {info.get('category')}",
            f"  Part ID / SKU:     {info.get('part_id')}",
            "-" * 60,
            "  Verified Sellers:"
        ]
        for s in info.get("sellers", []):
            card.append(f"    - {s['name']} (Rating: {s['rating']}/5.0, {s['location']}) - Delivery: {s['delivery_days']} days")
        card.append("=" * 60)
        return "\n".join(card)
