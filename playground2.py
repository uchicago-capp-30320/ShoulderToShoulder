class PanelEventViewSet(viewsets.ModelViewSet):
    queryset = PanelEvent.objects.all()
    serializer_class = PanelEventSerializer
    permission_classes = [HasAppToken]

    def get_queryset(self):
        queryset = self.queryset
        event_id = self.request.query_params.get('event_id')

        if event_id:
            user = User.objects.filter(id=event_id)
            if user:
                queryset = queryset.filter(event_id=user)

        return queryset

    def create(self, request, *args, **kwargs):
        # Ensure the event_id is provided in the request
        event_id = request.data.get('event_id')
        if not event_id:
            return Response({"error": "User ID not provided"}, status=400)
        
        # try to get the event
        try:
            event = Event.objects.get(event_id=event_id)
        except Event.DoesNotExist:
            return Response({"error": "Event not found"}, status=404)

        event_data = {"event_id": event}
        # Prepare panel event data data based on the user and their onboarding data
        try:
            paneled_events_lst = self.prepare_user_events(event_id, event_data)
            panel_events_objs = [PanelEvent(**panel_event) for panel_event in paneled_events_lst]
            PanelEvent.objects.bulk_create(panel_events_objs)
        except Exception as e:
            return Response({"error": f"Failed to create event suggestions: {str(e)}"}, status=400)

        # Return a success response
        return Response({"detail": "Successfully updated"}, status=201)

    def num_participant_mapping(self, value):
        """
        Helper function to standardize participant data into necessary format
        for EventSugestions Row

        Inputs:
            value (str): string of user's preferred number of event participants, if empty returns Null

       Return: string: String of the value parsed into the necessary format
        """

        if "1-5" in value:
            return "1to5"
        elif "5-10" in value:
            return "5to10"
        elif "10-15" in value:
            return "10to15"
        elif "15+" in value:
            return "15p"
        else:
            return None

    def prepare_user_events(self, event_id, template_event_data):
        """
        Helper function to prepare user events data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            event_id: User's ID
            template_event_data (dict): Dictionary of user's event suggestions data

        Returns: event_lst (lst): List of dictionaries containing all of a
        user's unique events and answers as well as onboarding information
        """
        # parse both event 1 and event 2 to a list of dictionaries that can be turned into rows in PanelEvent
        user_events = Event.objects.filter(event_id=event_id)

        event_lst = []

        for event in user_events:
            # copy the onboarding data into a new dictionary for the separate scenearios
            user_event_data_template = template_event_data.copy()
            if event.prefers_event1 is not None or event.prefers_event2 is not None:
                event_data = self.parse_event_data(event, user_event_data_template)

                event_lst.append(event_data)

        return event_lst

    def parse_event_data(self, event, data_template):
        """
        Helper function to parse user events data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            event (event): Rows of the user's events from the event Model
            data_template (dict): Dictionary of user's event suggestions data

        Returns:
            event_data: Event 1 with preferences and onboarding information
        """
        event_data = data_template.copy()
        event_data.update(self.parse_hobby_type(event.hobby_type))
        num_part = f"num_particip_{self.num_participant_mapping(event.max_attendees)}"
        event_data.update({num_part: True})
        event_data.update(self.parse_event_datetime(event.datetime))
        event_data.update(self.parse_duration(event.duration_h))

        return event_data

    def parse_hobby_type(self, hobby_type):
        """
        Helper function to parse user's event hobby type data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            hobby_type (Hobby): Hobby Type of user's event

        Returns:
            hobby_data (dict): dictionary of user's event hobby type
        """
        hobby_data = {}

        category_mapping = {
            "TRAVEL": "hobby_category_travel",
            "ARTS AND CULTURE": "hobby_category_arts_and_culture",
            "LITERATURE": "hobby_category_literature",
            "FOOD AND DRINK": "hobby_category_food",
            "COOKING/BAKING": "hobby_category_cooking_and_baking",
            "SPORT/EXERCISE": "hobby_category_exercise",
            "OUTDOORS":  "hobby_category_outdoor_activities",
            "CRAFTING": "hobby_category_crafting",
            "HISTORY AND LEARNING": "hobby_category_history",
            "COMMUNITY EVENTS": "hobby_category_community",
            "GAMING": "hobby_category_gaming",
        }

        for hobby_key in category_mapping.values():
            hobby_data[hobby_key] = False
        
        hobby_data[category_mapping[hobby_type.type]] = True

        return hobby_data

    def parse_event_datetime(self, datetime):
        """
        Helper function to parse user's event date and time data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            datetime (datetime): datetime of user's event

        Returns:
            event_datetime_mapping (dict): dictionary of user's event day and time data
        """
        event_datetime_mapping = {}
        days_of_week = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        time_periods = ["early_morning", "morning", "afternoon", "evening", "night", "late_night"]
        
        # convert datetime to day of week and time of day
        day_of_week = datetime.strftime('%A').lower()
        tod_string = self.get_time_of_day(datetime.hour)
        if tod_string is None:
            return event_datetime_mapping
        tod_standardized = "_".join(tod_string.lower().split())

        # create a dictionary with all the days of the week and time periods
        for day in days_of_week:
            for period in time_periods:
                field_name = f"{day}_{period}"
                if field_name == f"{day_of_week}_{tod_standardized}":
                    event_datetime_mapping[field_name] = True
                else:
                    event_datetime_mapping[field_name] = False

        return event_datetime_mapping

    def get_time_of_day(self, hour):
        time_period_mapping = {
            'early_morning': [5, 6, 7, 8],
            'morning': [9, 10, 11, 12],
            'afternoon': [13, 14, 15, 16],
            'evening': [17, 18, 19, 20],
            'night': [21, 22, 23, 24],
            'late_night': [1, 2, 3, 4]
        }

        for time_period, hours in time_period_mapping.items():
            if hour in hours:
                return time_period
        
        return None

    def parse_duration(self, duration):
        """
        Helper function to parse user's event duration data formatting and standardizing
        it into the PanelEvent row

        Inputs:
            duration (event): duration of user's event

        Returns:
            duration_data (dict): dictionary of user's event duration data
        """
        duration_data = {
        f"duration_{i}hr": i == duration for i in range(1, 9)
    }
        return duration_data