"""Launch the RSNA campaign without eager orchestration imports in spawned workers."""


def main() -> int:
    """Import and run the campaign only in the parent entry process."""
    from radfusion.training.rsna_campaign import main as run_campaign

    return run_campaign()


if __name__ == "__main__":
    raise SystemExit(main())
