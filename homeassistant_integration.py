#!/usr/bin/env python3
"""
Home Assistant Integration for AI Assistant
Allows the AI to control and query Home Assistant devices via REST API
"""

import requests
import json
from typing import Dict, List, Any, Optional

class HomeAssistantClient:
    def __init__(self, base_url: str, access_token: str):
        """
        Initialize Home Assistant client
        
        Args:
            base_url: Home Assistant URL (e.g., "http://homeassistant.local:8123")
            access_token: Long-lived access token from Home Assistant
        """
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }
    
    def get_states(self) -> List[Dict[str, Any]]:
        """Get all entity states"""
        response = requests.get(
            f"{self.base_url}/api/states",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def get_entity_state(self, entity_id: str) -> Dict[str, Any]:
        """Get state of a specific entity"""
        response = requests.get(
            f"{self.base_url}/api/states/{entity_id}",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def call_service(self, domain: str, service: str, entity_id: str = None, **kwargs) -> Dict[str, Any]:
        """
        Call a Home Assistant service
        
        Args:
            domain: Service domain (e.g., "light", "switch", "automation")
            service: Service name (e.g., "turn_on", "turn_off", "toggle")
            entity_id: Optional entity ID to target
            **kwargs: Additional service data
        """
        data = kwargs.copy()
        if entity_id:
            data['entity_id'] = entity_id
        
        response = requests.post(
            f"{self.base_url}/api/services/{domain}/{service}",
            headers=self.headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def turn_on(self, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Turn on a device"""
        domain = entity_id.split('.')[0]
        return self.call_service(domain, "turn_on", entity_id, **kwargs)
    
    def turn_off(self, entity_id: str) -> Dict[str, Any]:
        """Turn off a device"""
        domain = entity_id.split('.')[0]
        return self.call_service(domain, "turn_off", entity_id)
    
    def toggle(self, entity_id: str) -> Dict[str, Any]:
        """Toggle a device"""
        domain = entity_id.split('.')[0]
        return self.call_service(domain, "toggle", entity_id)
    
    def get_lights(self) -> List[Dict[str, Any]]:
        """Get all light entities"""
        states = self.get_states()
        return [s for s in states if s['entity_id'].startswith('light.')]
    
    def get_switches(self) -> List[Dict[str, Any]]:
        """Get all switch entities"""
        states = self.get_states()
        return [s for s in states if s['entity_id'].startswith('switch.')]
    
    def get_sensors(self) -> List[Dict[str, Any]]:
        """Get all sensor entities"""
        states = self.get_states()
        return [s for s in states if s['entity_id'].startswith('sensor.')]
    
    def trigger_automation(self, entity_id: str) -> Dict[str, Any]:
        """Trigger an automation"""
        return self.call_service("automation", "trigger", entity_id)
    
    def send_notification(self, message: str, title: str = None, target: str = "notify") -> Dict[str, Any]:
        """Send a notification"""
        data = {"message": message}
        if title:
            data["title"] = title
        return self.call_service("notify", target, **data)


if __name__ == "__main__":
    import os
    
    # Example usage - configure with environment variables
    HA_URL = os.getenv('HA_URL', 'http://homeassistant.local:8123')
    ACCESS_TOKEN = os.getenv('HA_TOKEN', 'YOUR_TOKEN_HERE')
    
    ha = HomeAssistantClient(HA_URL, ACCESS_TOKEN)
    
    print("=== Home Assistant Integration Demo ===")
    print(f"Connected to: {HA_URL}\n")
    
    # List some devices
    print("Lights:")
    for light in ha.get_lights()[:5]:
        print(f"  {light['entity_id']}: {light['state']}")
