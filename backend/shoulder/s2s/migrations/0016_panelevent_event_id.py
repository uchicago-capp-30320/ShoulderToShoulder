# Generated by Django 5.0.4 on 2024-05-07 23:03

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("s2s", "0015_remove_panelevent_dist_within_10mi_and_more"),
    ]

    operations = [
        migrations.AddField(
            model_name="panelevent",
            name="event_id",
            field=models.ForeignKey(
                default=1, on_delete=django.db.models.deletion.CASCADE, to="s2s.event"
            ),
            preserve_default=False,
        ),
    ]
