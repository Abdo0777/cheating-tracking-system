from plyer import notification


def notify_cheating(track_id, behavior):
    try:
        notification.notify(
            title="Cheating Alert Detected",
            message=f"Student {track_id}\nBehavior: {behavior}",
            timeout=5,
        )
    except Exception as e:
        print(f"[notifier] could not send desktop notification: {e}")
