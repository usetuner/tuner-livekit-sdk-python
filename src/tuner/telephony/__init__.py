"""Carrier bridges that carry Tuner's simulation correlation header to LiveKit.

The agent side needs nothing from here: ``TunerPlugin`` reads the header from the SIP
participant. These helpers run in front of LiveKit, in a carrier's webhook.
"""
